"""
Download ML artifacts from Google Drive.
Run once before starting the backend.

Usage:
    pip install gdown
    python download_artifacts.py
"""
import subprocess, sys, os

FOLDER_ID   = "1FS27498ltu4seS844g6yBcO6sAXqSCCQ"
DEST_DIR    = os.path.join(os.path.dirname(__file__),
                           "backend", "infrastructure", "ml", "artifacts")

def install_gdown():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gdown>=4.7"])

def download():
    import gdown
    os.makedirs(DEST_DIR, exist_ok=True)
    print(f"Downloading artifacts → {DEST_DIR}")
    gdown.download_folder(
        id=FOLDER_ID,
        output=DEST_DIR,
        quiet=False,
        use_cookies=False,
    )
    print("Done.")

if __name__ == "__main__":
    try:
        import gdown
    except ImportError:
        print("Installing gdown...")
        install_gdown()
        import gdown
    download()
