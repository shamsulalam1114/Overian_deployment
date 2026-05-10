import os
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import cv2
import streamlit as st
from PIL import Image
from datetime import datetime
from fpdf import FPDF
from fpdf.enums import XPos, YPos

import torch
import torch.nn as nn
from torchvision import models, transforms

# -------------------------------------------------------
# CLASS INDEX ORDER must match torchvision ImageFolder (alphabetical folder names).
# Training notebook folders: Ovarian_Cancer, Ovarian_Non_Cancer
#   -> index 0 = Ovarian_Cancer (= malignant tissue)
#   -> index 1 = Ovarian_Non_Cancer (= benign / non-cancer slide)
# If your folders differ, reorder CLASS_NAMES accordingly.
# -------------------------------------------------------
CLASS_NAMES   = ['malignant', 'benign']
CLASS_DISPLAY = {
    'benign':    'Benign',
    'malignant': 'Malignant (cancer)',
}

# =====================================================
# MODEL ARCHITECTURE (must match training notebook)
# =====================================================
class CBAMBlock(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // ratio, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False)

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.mlp(self.avg_pool(x).view(b, c))
        max_out = self.mlp(self.max_pool(x).view(b, c))
        channel_w = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        x = x * channel_w
        avg_sp = torch.mean(x, dim=1, keepdim=True)
        max_sp = torch.max(x, dim=1, keepdim=True)[0]
        spatial_w = self.sigmoid(self.conv_spatial(torch.cat([avg_sp, max_sp], dim=1)))
        return x * spatial_w


class HybridAttnModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        base_effnet = models.efficientnet_v2_s(weights=None)
        self.effnet_features = base_effnet.features
        self.cbam = CBAMBlock(in_channels=1280)
        self.effnet_pool = nn.AdaptiveAvgPool2d(1)

        self.swin = models.swin_t(weights=None)
        swin_dim = self.swin.head.in_features
        self.swin.head = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(1280 + swin_dim, 512),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.7),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.7),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        eff_map  = self.effnet_features(x)
        refined  = self.cbam(eff_map)
        feat_loc = self.effnet_pool(refined).view(refined.size(0), -1)
        feat_glb = self.swin(x)
        return self.classifier(torch.cat((feat_loc, feat_glb), dim=1))


# =====================================================
# MODEL WEIGHTS (local file OR remote URL for cloud deploy)
# =====================================================
_LOCAL_WEIGHTS_NAME = "best_model_fold5.pth"


def _model_url_from_env_or_secrets() -> str:
    for key in ("MODEL_URL", "OVAPREDICT_MODEL_URL"):
        v = (os.environ.get(key) or "").strip()
        if v:
            return v
    try:
        cfg = getattr(st, "secrets", None)
        if cfg is not None:
            v = str(cfg.get("MODEL_URL", "") or "").strip()
            if v:
                return v
    except Exception:
        pass
    return ""


def _download_headers(url: str):
    h = {"User-Agent": "OvaPredict-streamlit/1.0"}
    if "huggingface.co" not in url.lower():
        return h
    tok = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if not tok:
        try:
            cfg = getattr(st, "secrets", None)
            if cfg is not None:
                tok = str(cfg.get("HF_TOKEN") or cfg.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
        except Exception:
            tok = ""
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


def _download_weights_to(url: str, dest: Path, meta: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink(missing_ok=True)
    req = urllib.request.Request(url, headers=_download_headers(url))
    try:
        with urllib.request.urlopen(req, timeout=1200) as resp:
            with open(tmp, "wb") as out:
                while True:
                    chunk = resp.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
        tmp.replace(dest)
        meta.write_text(url, encoding="utf-8")
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def resolve_model_weights_path() -> str:
    """
    Local: `best_model_fold5.pth` next to this file (dev / small deploy).

    Cloud / huge checkpoint: set `MODEL_URL` in Streamlit Secrets or the environment
    to a direct HTTPS link (e.g. Hugging Face `resolve/main/...` URL). Weights are
    cached under ``~/.cache/ovapredict/`` on the host.
    """
    app_dir = Path(__file__).resolve().parent
    local = app_dir / _LOCAL_WEIGHTS_NAME
    url = _model_url_from_env_or_secrets()

    if url:
        parsed = urllib.parse.urlparse(url)
        fname = os.path.basename(parsed.path) or _LOCAL_WEIGHTS_NAME
        if not fname.endswith(".pth"):
            fname = _LOCAL_WEIGHTS_NAME
        cache_dir = Path.home() / ".cache" / "ovapredict"
        cached = cache_dir / fname
        meta = cache_dir / (fname + ".source_url")

        need = True
        if cached.exists() and cached.stat().st_size > 4096 and meta.exists():
            try:
                need = meta.read_text(encoding="utf-8").strip() != url
            except OSError:
                need = True
        if need:
            with st.spinner(
                "Downloading model weights (large file — first run only; cached after that)…"
            ):
                _download_weights_to(url, cached, meta)
        return str(cached)

    if local.exists():
        return str(local)

    st.error("**No model checkpoint found.**")
    st.markdown(
        f"- Deploy: add **`MODEL_URL`** in [Streamlit Cloud Secrets](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/secrets-management) "
        f" pointing to your `.pth` (HTTPS), **or**\n"
        f"- Put **`{_LOCAL_WEIGHTS_NAME}`** next to `app.py`.\n\n"
        "Do **not** commit huge weights to GitHub; use Hugging Face Hub, cloud storage signed URL, or similar."
    )
    st.stop()


# =====================================================
# MODEL LOADING
# =====================================================
@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    try:
        model = HybridAttnModel(num_classes=2)
        try:
            state = torch.load(model_path, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state, strict=False)
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"**Fatal Error:** Model file not found at `{model_path}`.")
        st.stop()
    except Exception as e:
        st.error(f"**Fatal Error:** Could not load model — {e}")
        st.stop()


# =====================================================
# PREPROCESSING (matches val_transforms in notebook)
# =====================================================
_val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def preprocess(pil_img: Image.Image) -> torch.Tensor:
    return _val_tf(pil_img.convert('RGB')).unsqueeze(0)


# =====================================================
# CBAM ATTENTION HEATMAP (no backward pass required)
# Hooks into CBAMBlock.conv_spatial to read the spatial
# attention weights the model learned — 100% reliable.
# =====================================================
def compute_attention_cam(model: nn.Module, pil_img: Image.Image) -> np.ndarray:
    """
    Extract CBAM spatial attention map via a forward hook.
    Returns a (H, W) float32 numpy array normalised to [0, 1].
    """
    captured = {}

    def _hook(module, inp, out):
        attn = torch.sigmoid(out).detach().squeeze()  # (1, H, W) or (H, W)
        if attn.dim() == 3:
            attn = attn[0]
        captured['cam'] = attn.cpu().numpy()

    handle = model.cbam.conv_spatial.register_forward_hook(_hook)
    try:
        tensor = preprocess(pil_img)
        with torch.no_grad():
            feat = model.effnet_features(tensor)  # (1, 1280, H, W)
            refined = model.cbam(feat)             # triggers hook
    finally:
        handle.remove()

    cam = captured.get('cam')
    if cam is None:
        raise RuntimeError('CBAM hook captured nothing — check model architecture.')

    # Blend spatial attention with mean channel-activation for richer map
    energy = refined.squeeze(0).mean(0).cpu().numpy()  # (H, W)
    if energy.shape == cam.shape:
        e_norm = (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)
        cam = 0.5 * cam + 0.5 * e_norm

    if cam.max() != cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    return cam.astype(np.float32)


def jet_colormap(gray: np.ndarray) -> np.ndarray:
    """Pure-numpy JET colormap — returns (H,W,3) uint8."""
    t = np.clip(gray, 0, 1)
    r = np.clip(1.5 - abs(4*t - 3), 0, 1)
    g = np.clip(1.5 - abs(4*t - 2), 0, 1)
    b = np.clip(1.5 - abs(4*t - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def overlay_heatmap(pil_img: Image.Image, cam: np.ndarray) -> Image.Image:
    img_np  = np.array(pil_img.convert('RGB').resize((224, 224))).astype(float)
    cam_224 = np.array(Image.fromarray(cam).resize((224, 224),
                        resample=Image.BILINEAR)) if cam.shape != (224,224) \
              else cam
    heat    = jet_colormap(cam_224).astype(float)
    blend   = np.clip(0.55 * img_np + 0.45 * heat, 0, 255).astype(np.uint8)
    return Image.fromarray(blend)


def qualify_histopathology_candidate(pil_rgb: Image.Image) -> tuple[bool, list[str]]:
    """
    Cheap quality gate — NOT a learnt OOD model. Helps block books, selfies, screenshots, blank pages before
    the classifier’s benign/malignant output is surfaced as diagnostic-looking text.

    Yellow / chartreuse *print ink* often matched the naive RGB „eosin” mask; we veto those with HSV and
    require pink eosin (R clearly above G), plus at least modest hematoxylin-blue hue density for typical H&E.

    Tune thresholds on your cohort if you observe false rejects on valid tiles.
    """
    messages: list[str] = []
    rgb = pil_rgb.convert('RGB').resize((224, 224))
    rgb_u8 = np.asarray(rgb, dtype=np.uint8)
    arr = rgb_u8.astype(np.float64) / 255.0
    r = arr[..., 0]
    g = arr[..., 1]
    b = arr[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b

    gs_std = float(gray.std())
    g8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    lap_var = float(cv2.Laplacian(g8, cv2.CV_64F).var())

    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    h_raw = hsv[..., 0].astype(np.float32)
    sat = hsv[..., 1].astype(np.float32) / 255.0
    val = hsv[..., 2].astype(np.float32) / 255.0

    # Glossy ink yellow / lime / yellow-green typical of covers and packaging (OpenCV hue 0–179).
    chartreuse_cover = (
        (h_raw >= 16.0)
        & (h_raw <= 52.0)
        & (sat >= 0.26)
        & (val >= 0.30)
    )
    frac_chartreuse = float(np.mean(chartreuse_cover))

    # Purple / hematoxylin-blue band (helps ensure we are not *only* warm print colours).
    hema_blue_hue = (
        (h_raw >= 94.0)
        & (h_raw <= 152.0)
        & (sat >= 0.10)
        & (val >= 0.10)
    )
    frac_hema_hue = float(np.mean(hema_blue_hue))

    # Mustard / butter yellow where R≈G and B is clearly lower — still confuses RGB-only eosin tests.
    yellowish_print = (
        (r > 0.36)
        & (g > 0.36)
        & (np.abs(r - g) < 0.095)
        & (b < np.maximum(r, g) * 0.78)
    )
    # True eosin pink: warm R but R meaningfully above G (unlike yellow ink or wood midtones).
    eosin_pink = (
        ~yellowish_print
        & (r > 0.40)
        & (g > 0.22)
        & ((r - g) > 0.018)
        & (r > b + 0.028)
        & (b < 0.58)
    )
    eosin_like = float(np.mean(eosin_pink))

    nuclear_dark = float(np.mean((gray < 0.55) & ((b + 0.04 > r) | (gray < r * 1.08))))
    hema_like = float(
        np.mean((gray >= 0.08) & (gray <= 0.72) & (b + 0.02 >= r) & ((b + g) > r * 0.92))
    )
    colourful = float(np.std(arr, axis=2).mean())

    # Consumer-object veto: lots of saturated chartreuse ink and almost no hematoxylin-blue hue.
    consumer_print_veto = (frac_chartreuse > 0.095) and (frac_hema_hue < 0.045)

    stain_or_tissue_colour = (
        (
            eosin_like > 0.014
            or nuclear_dark > 0.038
            or hema_like > 0.050
            or colourful > 0.052
        )
        and (
            frac_hema_hue >= 0.024
            or eosin_like > 0.022
            or nuclear_dark > 0.046
            or hema_like > 0.060
        )
    )

    lap_not_too_smooth = lap_var >= 12.0

    loose_second_chance = (
        not consumer_print_veto
        and gs_std >= 0.042
        and frac_hema_hue >= 0.034
        and (eosin_like + nuclear_dark + 0.45 * hema_like) > 0.072
        and lap_not_too_smooth
    )

    qualified = (
        gs_std >= 0.0085
        and stain_or_tissue_colour
        and lap_not_too_smooth
        and not consumer_print_veto
    )
    if not qualified and loose_second_chance:
        qualified = True
        messages.append('Borderline input — treat model output cautiously.')

    if not qualified:
        if consumer_print_veto:
            messages.append(
                'Colour layout matches consumer print (yellow / chartreuse ink) more than H&E microscopy — '
                'typical of book covers, packaging, or posters.')
        if gs_std < 0.0085:
            messages.append(
                'Image has almost no contrast (flat colours, wash-out). Microscopy previews usually occupy more tonal range.')
        if lap_var < 12.0:
            messages.append(
                'Texture is too uniform or blurry versus typical histology magnification patterns.')
        if not stain_or_tissue_colour:
            messages.append(
                'Colours do not match routine H&E / tissue stain signatures (might be monochrome art, grayscale scans, unrelated photo).')

    return qualified, messages


# =====================================================
# PDF REPORT — Rich Clinical Report
# =====================================================
def _pdf_section(pdf, title):
    """Draw a bold blue section header."""
    pdf.set_font("Arial", "B", 12)
    pdf.set_fill_color(26, 58, 92)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 8, f"  {title}", ln=True, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)

def _pdf_row(pdf, col1, col2, widths=(70, 110), fill=False):
    """Draw a two-column table row."""
    pdf.set_fill_color(240, 244, 250)
    pdf.set_font("Arial", "", 9)
    pdf.cell(widths[0], 6, col1, border=1, fill=fill)
    pdf.cell(widths[1], 6, col2, border=1, fill=fill, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

def generate_pdf(pred_label: str, confidence: float,
                 orig_img: Image.Image, cam_img: Image.Image,
                 proba: np.ndarray) -> bytes:

    pdf = FPDF()
    pdf.add_page()
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── Cover Header ──
    pdf.set_fill_color(26, 58, 92)
    pdf.rect(0, 0, 210, 28, 'F')
    pdf.set_font("Arial", "B", 18)
    pdf.set_text_color(255, 255, 255)
    pdf.set_y(7)
    pdf.cell(0, 10, "OvaPredict AI - Ovarian Cancer Risk Report", ln=True, align="C")
    pdf.set_font("Arial", "", 9)
    pdf.set_text_color(180, 210, 240)
    report_id = datetime.now().strftime("%Y%m%d%H%M%S")
    pdf.cell(0, 6,
             f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}   |   "
             f"Report ID: OVA-{report_id}",
             ln=True, align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)

    # ── 1. Prediction Result ──
    _pdf_section(pdf, "1. AI Prediction Result")
    label_txt = "Malignant (Cancer)" if pred_label == 'malignant' else "Benign"
    color     = (220, 53, 69) if pred_label == 'malignant' else (40, 167, 69)
    pdf.set_font("Arial", "B", 16)
    pdf.set_text_color(*color)
    pdf.cell(0, 10, f"Diagnosis: {label_txt}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 7, f"Confidence Score: {confidence:.2f}%", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    # Confidence bar
    bar_w = 160
    fill_w = int(bar_w * confidence / 100)
    pdf.set_draw_color(200, 200, 200)
    pdf.set_fill_color(220, 220, 220)
    pdf.rect(15, pdf.get_y(), bar_w, 6, 'FD')
    pdf.set_fill_color(*color)
    pdf.rect(15, pdf.get_y(), fill_w, 6, 'F')
    pdf.ln(10)

    # ── 2. Class Probability Table ──
    _pdf_section(pdf, "2. Class Probability Breakdown")
    pdf.set_font("Arial", "B", 9)
    pdf.set_fill_color(220, 230, 242)
    pdf.cell(70, 6, "Class",      border=1, fill=True)
    pdf.cell(60, 6, "Probability", border=1, fill=True)
    pdf.cell(50, 6, "Interpretation", border=1, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Arial", "", 9)
    class_interp = ["Non-cancerous tissue", "Cancerous / Malignant tissue"]
    for i, cls in enumerate(CLASS_NAMES):
        pct = float(proba[i]) * 100
        is_pred = (i == (1 if pred_label == 'malignant' else 0))
        pdf.set_font("Arial", "B" if is_pred else "", 9)
        pdf.cell(70, 6, cls.title() + (" <-- Predicted" if is_pred else ""), border=1)
        pdf.cell(60, 6, f"{pct:.2f}%", border=1)
        pdf.cell(50, 6, class_interp[i], border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Arial", "", 9)
    pdf.ln(4)

    # ── 3. Risk Interpretation Table ──
    _pdf_section(pdf, "3. Risk Interpretation & Clinical Indicators")
    if pred_label == 'malignant':
        risk_rows = [
            ("Overall Risk Level",     "HIGH - Malignant features detected"),
            ("Classification",         "Malignant (Cancerous) Ovarian Mass"),
            ("Histological Pattern",   "Abnormal cellular architecture detected by model"),
            ("Recommended Action",     "Immediate specialist referral required"),
            ("Follow-up",              "Oncology consultation, staging workup, biopsy confirmation"),
            ("Confidence Threshold",   f"{confidence:.1f}% (High confidence prediction)"
                                       if confidence >= 80 else
                                       f"{confidence:.1f}% (Moderate - confirm with biopsy)"),
        ]
    else:
        risk_rows = [
            ("Overall Risk Level",     "LOW - Benign features detected"),
            ("Classification",         "Benign (Non-Cancerous) Ovarian Mass"),
            ("Histological Pattern",   "Normal / benign cellular architecture detected"),
            ("Recommended Action",     "Routine monitoring recommended"),
            ("Follow-up",              "Periodic ultrasound follow-up, gynecologist review"),
            ("Confidence Threshold",   f"{confidence:.1f}% (High confidence prediction)"
                                       if confidence >= 80 else
                                       f"{confidence:.1f}% (Borderline - additional imaging advised)"),
        ]
    pdf.set_font("Arial", "B", 9)
    pdf.set_fill_color(220, 230, 242)
    pdf.cell(70, 6, "Indicator",     border=1, fill=True)
    pdf.cell(110, 6, "Finding",      border=1, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    for i, (k, v) in enumerate(risk_rows):
        fill = (i % 2 == 0)
        pdf.set_fill_color(245, 248, 252) if fill else pdf.set_fill_color(255, 255, 255)
        pdf.set_font("Arial", "B", 9)
        pdf.cell(70, 6, k, border=1, fill=fill)
        pdf.set_font("Arial", "", 9)
        pdf.cell(110, 6, v, border=1, fill=fill, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    # ── 4. Images ──
    _pdf_section(pdf, "4. Input Image & CBAM Attention Heatmap")
    pdf.set_font("Arial", "", 8)
    pdf.cell(90, 5, "Original Image", align="C")
    pdf.cell(90, 5, "Attention Heatmap Overlay", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    base      = os.path.dirname(os.path.abspath(__file__))
    orig_path = os.path.join(base, "_tmp_orig.png")
    cam_path  = os.path.join(base, "_tmp_cam.png")
    orig_img.convert('RGB').resize((300, 300)).save(orig_path)
    cam_img.convert('RGB').resize((300, 300)).save(cam_path)

    y_img = pdf.get_y()
    pdf.image(orig_path, x=15,  y=y_img, w=85)
    pdf.image(cam_path,  x=108, y=y_img, w=85)
    pdf.set_y(y_img + 88)

    for p in [orig_path, cam_path]:
        try: os.remove(p)
        except: pass

    pdf.set_font("Arial", "I", 7)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, "Heatmap: Red/warm = high attention regions. Blue/cool = low attention.", ln=True, align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    # ── 5. Model Information ──
    _pdf_section(pdf, "5. AI Model Information")
    model_rows = [
        ("Model Architecture",  "HybridAttnModel: EfficientNetV2-S + Swin Transformer-T"),
        ("Attention Module",    "CBAM (Channel + Spatial Convolutional Block Attention)"),
        ("Training Method",     "5-Fold Stratified Cross-Validation"),
        ("Reported Accuracy",   "~99% Validation Accuracy per fold"),
        ("Classes",             "Benign / Malignant (2-class binary classification)"),
        ("Heatmap Method",      "CBAM Spatial Attention Map (forward-hook extraction)"),
        ("Framework",           "PyTorch + Streamlit"),
    ]
    pdf.set_font("Arial", "B", 9)
    pdf.set_fill_color(220, 230, 242)
    pdf.cell(70, 6, "Property",  border=1, fill=True)
    pdf.cell(110, 6, "Detail",   border=1, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    for i, (k, v) in enumerate(model_rows):
        fill = (i % 2 == 0)
        pdf.set_fill_color(245, 248, 252) if fill else pdf.set_fill_color(255, 255, 255)
        pdf.set_font("Arial", "B", 9)
        pdf.cell(70,  6, k, border=1, fill=fill)
        pdf.set_font("Arial", "", 9)
        pdf.cell(110, 6, v, border=1, fill=fill, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    # ── 6. Clinical Recommendations ──
    _pdf_section(pdf, "6. Clinical Recommendations")
    pdf.set_font("Arial", "", 9)
    if pred_label == 'malignant':
        recs = [
            "1. Refer immediately to a gynecologic oncologist for evaluation.",
            "2. Confirm diagnosis via histopathological biopsy before treatment.",
            "3. Order staging workup: CT/MRI of abdomen/pelvis, CA-125 serum marker.",
            "4. Multidisciplinary tumor board review is strongly recommended.",
            "5. Do NOT rely solely on this AI output for clinical decisions.",
        ]
    else:
        recs = [
            "1. Schedule routine follow-up ultrasound in 3-6 months.",
            "2. Review with a gynecologist if symptoms persist or worsen.",
            "3. Monitor CA-125 levels if clinically indicated.",
            "4. Annual pelvic examination as per standard screening guidelines.",
            "5. Do NOT rely solely on this AI output for clinical decisions.",
        ]
    for rec in recs:
        pdf.cell(0, 6, rec, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    # ── Footer ──
    pdf.set_font("Arial", "I", 7)
    pdf.set_text_color(120, 120, 120)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(2)
    pdf.cell(0, 5,
             "OvaPredict AI - For research & clinical decision support only. "
             "NOT a substitute for professional medical diagnosis. "
             "Always consult a qualified physician.",
             ln=True, align="C")
    pdf.set_text_color(0, 0, 0)
    return bytes(pdf.output())


# =====================================================
# CLINICAL REFERENCE
# =====================================================
CLINICAL = {
    "What Is Ovarian Cancer?": {
        "color": "#E8F4FF",
        "link": "https://gco.iarc.fr/today/data/factsheets/cancers/25-Ovary-fact-sheet.html",
        "text": [
            "Ovarian cancer usually denotes malignant epithelial tumours arising in or on the ovaries; adnexal masses may also prove benign, borderline, metastatic from other primaries, or non-epithelial. Prognosis is strongly subtype- and stage-dependent.",
            "The International Agency for Research on Cancer (IARC), which forms part of the World Health Organization, publishes rolling GLOBOCAN incidence and mortality tables. Prefer the linked fact sheet for dated numbers instead of quoting static pamphlet figures.",
            "There is still no ovarian screening programme with performance and cost-effectiveness comparable to organised cervical or breast screening for average-risk, asymptomatic people; evaluation of an adnexal mass typically combines history, exam, ultrasound, sometimes serum markers, and tissue when indicated.",
        ],
    },
    "Benign vs Malignant Ovarian Masses": {
        "color": "#E8FFF3",
        "link": "https://www.cancer.org/cancer/types/ovarian-cancer.html",
        "text": [
            "Benign entities include functional cysts, mature cystic teratomas (often called dermoid cysts), fibromas, endometriomas, and others. Borderline tumours are a distinct WHO category with specific management pathways.",
            "Invasive epithelial carcinomas are the dominant malignant group in adults; high-grade serous carcinoma is the prototype often discussed in public education, though multiple histologic types exist.",
            "Microscopic examination of tissue remains the reference for diagnosis; transvaginal ultrasound is the usual first imaging modality to characterise masses, supplemented by MRI or CT depending on suspicion and operative planning.",
        ],
    },
    "About This Application's Model": {
        "color": "#FFF4E6",
        "link": "",
        "text": [
            "The bundled architecture fuses EfficientNetV2-S convolutional features (with CBAM attention on the bottleneck map), pooled local descriptors, and parallel Swin-T global tokens prior to the linear classifier heads described in training notebook parity checks.",
            "The notebook uses five-fold stratified cross-validation; logged runs approached ~99% validation accuracy per fold on that internal ImageFolder split (class folders named Ovarian_Cancer and Ovarian_Non_Cancer in the author's Colab path). External cohort performance is unknown from this repo alone.",
            "Any assertion of bedside utility requires independent adjudication datasets, pathology review, drift monitoring, and regulatory alignment - not reproducible solely from retrospective fold accuracy.",
        ],
    },
    "Saliency Maps: CBAM in This App, Grad-CAM++ in the Literature": {
        "color": "#FFF1F2",
        "link": "https://arxiv.org/abs/1710.11063",
        "text": [
            "Rendered overlays combine CBAM spatial attention with mean channel activation (forward-only; no gradients). Pseudo-colour uses a jet-style ramp; hotter colours emphasize higher fused attention magnitude, which is correlation with convolutional suppression factors, not a calibrated biological boundary.",
            "Grad-CAM++ (Chattopadhyay et al., arXiv:1710.11063) is a complementary, gradient-driven explanation family for CNNs; it is cited for readers migrating from oncology literature referencing CAM-style tooling even though runtime heatmaps here are CBAM-derived.",
            "Heatmaps facilitate debugging model behaviour; definitive malignancy assessments always require microscopic diagnosis and multidisciplinary correlation.",
        ],
    },
}


# =====================================================
# PAGE CONFIG & STYLED UI
# =====================================================
st.set_page_config(
    page_title="OvaPredict",
    layout="wide",
    page_icon="🩺",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
.main .block-container { padding-left: 3rem !important; padding-right: 3rem !important; padding-top: 2rem !important; padding-bottom: 2rem !important; }
.stApp { background-color: #f4f7fb !important; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important; }

.header-banner {
    background: linear-gradient(135deg, #0a1628 0%, #132f4c 42%, #1a4a7c 100%);
    border-radius: 12px;
    padding: 32px 36px 28px 36px;
    margin-bottom: 28px;
    box-shadow: 0 4px 24px rgba(10, 22, 40, 0.18);
    text-align: center;
    border: 1px solid rgba(255,255,255,0.06);
}
.header-banner .header-eyebrow {
    color: rgba(255,255,255,0.55) !important;
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.16em !important;
    text-transform: uppercase !important;
    margin: 0 0 12px 0 !important;
}
.header-banner h1 {
    color: #ffffff !important;
    margin: 0 !important;
    font-size: 1.85rem !important;
    font-weight: 700 !important;
    text-align: center !important;
    letter-spacing: -0.03em;
}
.header-banner .header-sub {
    color: rgba(168, 212, 245, 0.92) !important;
    margin: 12px 0 0 0 !important;
    font-size: 0.9rem !important;
    line-height: 1.5 !important;
    max-width: 52rem;
    margin-left: auto !important;
    margin-right: auto !important;
}
.header-banner .header-meta {
    color: rgba(255,255,255,0.45) !important;
    margin: 16px 0 0 0 !important;
    font-size: 0.75rem !important;
}

.pred-col-title {
    font-weight: 600;
    font-size: 0.72rem;
    margin: 0 0 10px 0;
    color: #5c6b7a !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.card-benign, .card-malignant {
    border-radius: 14px;
    padding: 22px 20px;
    font-size: 1.6rem;
    font-weight: 700;
    text-align: center;
}
.card-benign {
    background: linear-gradient(135deg, #155724, #28a745);
    color: #ffffff !important;
    box-shadow: 0 6px 20px rgba(40,167,69,0.35);
    margin-bottom: 16px;
}
.card-malignant {
    background: linear-gradient(135deg, #7b0a18, #dc3545);
    color: #ffffff !important;
    box-shadow: 0 6px 20px rgba(220,53,69,0.35);
    margin-bottom: 16px;
}
.card-benign span, .card-malignant span {
    color: #ffffff !important;
    font-size: 1rem;
    font-weight: 400;
    display: block;
    margin-top: 6px;
}

.prob-bar-wrap { background: #dde4ef; border-radius: 8px; height: 18px; width: 100%; margin: 4px 0 14px 0; }
.prob-bar-fill { height: 18px; border-radius: 8px; transition: width 0.5s ease; }

.upload-zone {
    background: #ffffff;
    border: 2.5px dashed #3a7bd5;
    border-radius: 16px;
    padding: 52px 32px;
    text-align: center;
}
.upload-zone h3 {
    color: #1a3a5c !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
}
.upload-zone p { color: #5c6b7a !important; font-size: 0.9rem !important; }

.stTabs [data-baseweb="tab"] { font-size: 1rem !important; font-weight: 600 !important; }
[data-testid="stDownloadButton"] button {
    border-radius: 10px !important;
    border: none !important;
    background: linear-gradient(135deg, #1a3a5c, #2460a7) !important;
    font-weight: 600 !important;
    width: 100%;
}
[data-testid="stDownloadButton"] button:hover {
    background: linear-gradient(135deg, #2460a7, #3a7bd5) !important;
}

.ref-card {
    border: 1px solid rgba(0,0,0,0.07);
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 18px;
}
.ref-card h4 { color: #2d4a6e !important; margin: 0 0 8px 0; font-size: 1.05rem !important; }
.ref-card p { font-size: 0.95rem; line-height: 1.65; margin: 5px 0; color: #2d3748 !important; }
.ref-card a { color: #2460a7 !important; font-weight: 600; text-decoration: none; }

.disclaimer {
    background: #fff8e1;
    border-left: 4px solid #f59e0b;
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 0.88rem;
    color: #78350f !important;
    margin-top: 16px;
}

.pred-results .prob-row { margin: 0 0 12px 0; font-size: 0.9rem; color: #1a1a2e; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="header-banner">
    <p class="header-eyebrow">Ovarian tissue · AI-assisted triage</p>
    <h1>OvaPredict</h1>
    <p class="header-sub">Uploads must be <b>ovarian H&amp;E histopathology</b> only (microscope tiles from this study).
    The model scores benign vs malignant signal on that domain — EfficientNetV2-S, Swin-T, and CBAM, 5-fold CV.</p>
    <p class="header-meta">Do not upload photos, books, whole-body imaging, or non-ovarian slides. Decision support only.</p>
</div>
""",
    unsafe_allow_html=True,
)


# =====================================================
# MODEL LOAD
# =====================================================
with st.spinner("Loading neural network checkpoint (may take up to ~30 seconds on first run)…"):
    model = load_model(resolve_model_weights_path())


# =====================================================
# TABS
# =====================================================
tab_pred, tab_ref = st.tabs(["Analysis", "Clinical reference"])


# ─────────────────────────────────────────────
# TAB 1 — PREDICTION
# ─────────────────────────────────────────────
with tab_pred:
    st.subheader("Ovarian histopathology upload")
    st.caption(
        "Accepted formats: JPEG, PNG, TIFF, BMP. "
        "Use a single de-identified H&E crop from ovarian tissue (training scope for this model)."
    )

    st.warning(
        "**This tool accepts ovarian H&E histopathology images only.** "
        "Photographs, phone pictures of books, screenshots, and other organs or stains are not valid inputs."
    )

    uploaded = st.file_uploader(
        "Ovarian H&E histopathology image (required)",
        type=["jpg", "jpeg", "png", "tiff", "tif", "bmp"],
        help="One microscope-style tile: hematoxylin & eosin, ovarian tissue. Not general photos.",
        label_visibility="visible",
        key="ovapredict_upload",
    )

    confirmed = st.checkbox(
        "I confirm this upload is ovarian tissue on an H&E-stained microscopy slide "
        "(a pathology crop/tile — not a photo of something else).",
        value=False,
        key="confirm_ovarian_he_only",
    )

    if uploaded and not confirmed:
        uploaded.seek(0)
        preview = Image.open(uploaded).convert("RGB")
        st.info("Check the confirmation box above to run analysis on this file.")
        st.image(preview.resize((320, 320)), width=320)
    elif uploaded and confirmed:
        uploaded.seek(0)
        pil_img = Image.open(uploaded).convert('RGB')
        qualifies, qualify_notes = qualify_histopathology_candidate(pil_img)

        with st.spinner("Running inference…"):
            tensor = preprocess(pil_img)
            with torch.no_grad():
                logits = model(tensor)
                proba = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred_idx = int(np.argmax(proba))
            pred_label = CLASS_NAMES[pred_idx]
            confidence = float(proba[pred_idx]) * 100

        cam_img = pil_img
        if qualifies:
            try:
                cam_np = compute_attention_cam(model, pil_img)
                cam_img = overlay_heatmap(pil_img, cam_np)
            except Exception as e:
                st.error(f"Heatmap generation failed: {e}")

        if qualifies and qualify_notes:
            for note in qualify_notes:
                st.info(note)

        if not qualifies:
            st.error(
                "**Not accepted as ovarian H&E microscopy.** "
                "This app only evaluates crops that look like ovarian "
                "**hematoxylin & eosin** tissue from a microscope. Photos, printed pages, screenshots, "
                "other stains, or other organs cannot be analysed here."
            )
            for reason in qualify_notes:
                st.markdown(f"- {reason}")
            with st.expander("Raw softmax output — debugging only — not pathology advice", expanded=False):
                st.write({CLASS_NAMES[i]: float(proba[i]) for i in range(len(CLASS_NAMES))})

        col_orig, col_heat, col_res = st.columns([1, 1, 1.2], gap="medium")

        prob_rows_html = "".join(
            (
                '<div class="prob-row"><small><b>'
                f"{cls.title()}</b>: {float(proba[i]) * 100:.2f}%</small>"
                '<div class="prob-bar-wrap"><div class="prob-bar-fill" '
                f'style="width:{float(proba[i]) * 100:.1f}%;'
                f'background:{"#dc3545" if cls == "malignant" else "#28a745"};"></div></div></div>'
            )
            for i, cls in enumerate(CLASS_NAMES)
        )

        with col_orig:
            st.markdown(
                '<p class="pred-col-title">Input image</p>',
                unsafe_allow_html=True,
            )
            st.image(pil_img.resize((220, 220)), width=220)

        with col_heat:
            st.markdown(
                '<p class="pred-col-title">Saliency overlay (CBAM)</p>',
                unsafe_allow_html=True,
            )
            if qualifies:
                st.image(cam_img.resize((220, 220)), width=220)
                st.caption(
                    "Jet palette: warmer tones indicate stronger fused CBAM spatial attention weights "
                    "(interpretability only; not pathology ground truth)."
                )
            else:
                st.info(
                    "Heatmap withheld until the image clears screening as ovarian H&E-style microscopy.",
                )

        with col_res:
            st.markdown(
                '<p class="pred-col-title">Model output</p>',
                unsafe_allow_html=True,
            )
            if qualifies:
                card_cls = "card-malignant" if pred_label == "malignant" else "card-benign"
                label_d = "Malignant (Cancer)" if pred_label == "malignant" else "Benign"
                st.markdown(
                    '<div class="'
                    + card_cls
                    + '" style="font-size:1.15rem;padding:14px 10px;">'
                    + label_d
                    + f'<span style="font-size:0.85rem;">Confidence: {confidence:.2f}%</span></div>',
                    unsafe_allow_html=True,
                )
                st.markdown("**Class probabilities**")
                st.markdown(
                    '<div class="pred-results">' + prob_rows_html + "</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.info(
                    "No ovarian benign–malignant read is shown until screening accepts the "
                    "upload as ovarian H&E histopathology.",
                )

        if qualifies:
            try:
                pdf_bytes = generate_pdf(pred_label, confidence, pil_img, cam_img, proba)
                st.download_button(
                    "Download PDF report",
                    data=pdf_bytes,
                    file_name=f"OvaPredict_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    type="primary",
                )
            except Exception as e:
                st.error(f"PDF generation error: {e}")

        st.markdown(
            '<div class="disclaimer"><b>Disclaimer.</b> OvaPredict is intended for '
            '<b>research</b> and <b>decision-support</b> workflows. It does not constitute a '
            "medical device claim or substitute for pathology review. Align any action with institutional "
            "protocols and supervising clinicians.</div>",
            unsafe_allow_html=True,
        )

    else:
        st.markdown(
            """
        <div class="upload-zone">
            <h3>Upload ovarian H&E histopathology only</h3>
            <p>Select one microscopy tile above (drag-and-drop supported).</p>
            <p>JPEG, PNG, TIFF, or BMP · confirm the ovarian H&amp;E checkbox before analysis runs.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────
# TAB 2 — CLINICAL INTERPRETATION
# ─────────────────────────────────────────────
with tab_ref:
    st.header("Clinical Reference — Ovarian Cancer")
    st.markdown(
        "Evidence-based information on ovarian cancer, imaging, and the AI model used in OvaPredict."
    )
    st.markdown("---")

    for title, info in CLINICAL.items():
        paragraphs = "".join(f"<p>{p}</p>" for p in info["text"])
        link_block = ""
        if info["link"]:
            link_block = (
                '<div style="text-align:right;margin-top:10px;">'
                f'<a href="{info["link"]}" target="_blank" rel="noopener noreferrer" '
                'style="color:#2460a7;font-weight:600;">View external reference</a></div>'
            )
        st.markdown(
            f'<div class="ref-card" style="background:{info["color"]};">'
            f"<h4>{title}</h4>{paragraphs}{link_block}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        "<p style='text-align:center;color:#888;font-size:0.8rem;'>"
        "OvaPredict AI · EfficientNetV2-S + Swin-T + CBAM · 5-Fold CV · "
        "For Research & Clinical Decision Support Only</p>",
        unsafe_allow_html=True,
    )
