# OvaPredict Streamlit Deployment

This repository contains a Streamlit app for ovarian histopathology image triage:

- Main app: `app.py`
- Python dependencies: `requirements.txt`
- Optional local model file: `best_model_fold5.pth`

## What each file does

- `app.py`
  - Defines the hybrid PyTorch model (`EfficientNetV2-S + Swin-T + CBAM`)
  - Loads model weights from either:
    - local `best_model_fold5.pth`, or
    - remote `MODEL_URL` (recommended for cloud hosting)
  - Runs image qualification + inference
  - Generates CBAM heatmap overlays
  - Exports PDF reports
- `requirements.txt`
  - Lists Python packages needed by the app runtime
- `best_model_fold5.pth`
  - Trained checkpoint (local option). For Streamlit Cloud, prefer remote hosting with `MODEL_URL`.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create a new app and select:
   - Repository: this repo
   - Branch: your target branch
   - Main file path: `app.py`
4. In app settings, add secrets (recommended):
   - `MODEL_URL = "https://.../best_model_fold5.pth"`
   - Optional (if Hugging Face private file): `HF_TOKEN = "hf_..."`
5. Deploy.

## Model hosting recommendation

Do **not** rely on committing a very large `.pth` file to GitHub. Use a remote HTTPS model URL:

- Hugging Face (public/private with token)
- Cloud object storage signed URL
- Any stable direct-download HTTPS endpoint

The app caches downloaded weights under `~/.cache/ovapredict/` in the runtime environment.

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- Temporary files `_tmp_orig.png` and `_tmp_cam.png` are created only during PDF generation and then removed.
- If no local model exists and no `MODEL_URL` secret/environment variable is set, app startup stops with a clear error.
