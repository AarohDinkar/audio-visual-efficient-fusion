"""
Download MSR-VTT dataset for video captioning from Kaggle.

Uses kagglehub to download vishnutheepb/msrvtt.
Copies videos to data/videos/ and annotations to data/ for preprocessing.
"""

import json
import shutil
import sys
from pathlib import Path
from typing import Optional

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, VIDEOS_DIR


def download_from_kaggle() -> Optional[Path]:
    """
    Download MSR-VTT from Kaggle using kagglehub.

    Requires: pip install kagglehub
    Kaggle API: Set up ~/.kaggle/kaggle.json with your API credentials.
    """
    try:
        import kagglehub

        print("  Downloading MSR-VTT from Kaggle (vishnutheepb/msrvtt)...")
        path = kagglehub.dataset_download("vishnutheepb/msrvtt")
        print(f"  Downloaded to: {path}")
        return Path(path)
    except ImportError:
        print("  kagglehub not installed. Run: pip install kagglehub")
        return None
    except Exception as e:
        print(f"  Kaggle download failed: {e}")
        print("  Ensure you have Kaggle API credentials in ~/.kaggle/kaggle.json")
        return None


def setup_from_kaggle_path(kaggle_path: Path) -> bool:
    """
    Copy videos and annotations from Kaggle download to project data/ folder.

    Handles common MSR-VTT structures:
    - videos/ or TrainValVideo/ or similar
    - train_val_videodatainfo.json, MSRVTT_data.json, or other .json
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    # Find all video files
    video_exts = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
    video_files = []
    for ext in video_exts:
        video_files.extend(kaggle_path.rglob(f"*{ext}"))

    if not video_files:
        print("  No video files found in downloaded dataset.")
        return False

    print(f"  Found {len(video_files)} videos. Copying to {VIDEOS_DIR}...")
    for i, src in enumerate(video_files):
        dest = VIDEOS_DIR / src.name
        if not dest.exists() or dest.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dest)
        if (i + 1) % 500 == 0:
            print(f"    Copied {i + 1}/{len(video_files)}...")

    # Find annotation JSON files - copy to data/ and ensure train_val_videodatainfo.json exists
    json_files = list(kaggle_path.rglob("*.json"))
    annotations_copied = False
    primary_annot = DATA_DIR / "train_val_videodatainfo.json"
    for jf in json_files:
        dest = DATA_DIR / jf.name
        if jf.name in ("train_val_videodatainfo.json", "MSRVTT_data.json", "videodatainfo.json"):
            shutil.copy2(jf, dest)
            print(f"  Copied annotations: {jf.name} -> {dest}")
            if not primary_annot.exists():
                shutil.copy2(jf, primary_annot)
                print(f"  Also saved as {primary_annot.name} for preprocess compatibility")
            annotations_copied = True
        elif not annotations_copied:
            shutil.copy2(jf, dest)
            if not primary_annot.exists():
                shutil.copy2(jf, primary_annot)
            print(f"  Copied annotations: {jf.name} -> {dest}")
            annotations_copied = True

    return True


def download_msrvtt_sample():
    """Create sample structure if Kaggle download fails."""
    sample_dir = DATA_DIR / "msrvtt_sample"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_annotations = {
        "videos": [{"video_id": "video0", "split": "train", "captions": ["A dog running in a grassy field while barking."]}],
        "info": "Sample structure - add videos to data/videos/ and run preprocess_video.py",
    }
    with open(sample_dir / "sample_annotations.json", "w") as f:
        json.dump(sample_annotations, f, indent=2)
    print(f"  Created sample structure at {sample_dir}")


def main():
    """Main download routine."""
    print("MSR-VTT Dataset Setup (Kaggle)")
    print("=" * 40)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    # Download from Kaggle
    kaggle_path = download_from_kaggle()
    if kaggle_path and kaggle_path.exists():
        if setup_from_kaggle_path(kaggle_path):
            print("\nDone! Dataset ready.")
            print("Next: python scripts/preprocess_video.py --limit 100")
            return

    # Fallback: sample structure
    print("\nCreating sample structure for demo/testing...")
    download_msrvtt_sample()
    print("\nFor full dataset: Install kagglehub, set up Kaggle API, and run again.")
    print("  pip install kagglehub")
    print("  Place ~/.kaggle/kaggle.json with your Kaggle API credentials")
    print("\nFor quick demo: Place any .mp4/.avi video in data/videos/")
    print("Then run: python scripts/preprocess_video.py")


if __name__ == "__main__":
    main()
