"""
Download dataset and ML artifacts from Google Drive.
Run once before starting the backend.

Usage:
    pip install gdown
    python download_artifacts.py
"""
import subprocess, sys, os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Google Drive folders ──────────────────────────────────────────────────────
# Raw image dataset (screenshots organized by subject)
DATASET_FOLDER_ID = "1FS27498ltu4seS844g6yBcO6sAXqSCCQ"
DATASET_DEST      = os.path.join(ROOT, "data", "raw")

# ML artifacts (.pt model weights + graph data) — upload separately to Drive
# ARTIFACTS_FOLDER_ID = "TODO: upload .pt files to Drive, paste folder ID here"
# ARTIFACTS_DEST      = os.path.join(ROOT, "backend", "infrastructure", "ml", "artifacts")

def install_gdown():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gdown>=4.7"])

def download_folder(folder_id, dest, label):
    import gdown
    os.makedirs(dest, exist_ok=True)
    print(f"\n[{label}] Downloading -> {dest}")
    gdown.download_folder(
        id=folder_id,
        output=dest,
        quiet=False,
        use_cookies=False,
    )
    print(f"[{label}] Done.")

if __name__ == "__main__":
    try:
        import gdown
    except ImportError:
        print("Installing gdown...")
        install_gdown()
        import gdown

    download_folder(DATASET_FOLDER_ID, DATASET_DEST, "Raw dataset")

    # Uncomment after uploading .pt files to Drive:
    # download_folder(ARTIFACTS_FOLDER_ID, ARTIFACTS_DEST, "ML artifacts")

    print("\nSetup complete.")
    print("NOTE: ML artifacts (.pt files) must be placed in:")
    print(f"  backend/infrastructure/ml/artifacts/GNN_single_v1/")
    print(f"  backend/infrastructure/ml/artifacts/GNN_dual_v1/")
    print(f"  backend/infrastructure/ml/artifacts/GNN_dual_v2/")
