"""
OpenPlaque Colab setup script.

Usage in a Colab notebook:

    !git clone https://github.com/pazzani/OpenPlaque.git /content/OpenPlaque || true
    !git -C /content/OpenPlaque pull
    %run /content/OpenPlaque/colab/setup.py

This script:
- installs requirements-colab.txt
- adds OpenPlaque/src to sys.path
- configures nnU-Net environment variables
- verifies expected Google Drive files
"""

import os
import sys
import subprocess
from pathlib import Path


REPO_DIR = Path("/content/OpenPlaque")
DRIVE_ROOT = Path("/content/drive/MyDrive")
OPENPLAQUE_DRIVE = DRIVE_ROOT / "OpenPlaque"

FULL_DICOM_ZIP = OPENPLAQUE_DRIVE / "Full_DICOM.zip"
NNUNET_RESULTS_DIR = Path("/content/nnUNet_results")
NNUNET_RESULTS_ZIP = OPENPLAQUE_DRIVE / "models" / "Dataset001_CCTA_DHM-20260703T233210Z-3-001.zip"


def run(cmd):
    print("+", cmd)
    subprocess.run(cmd, shell=True, check=True)


def install_requirements():
    req = REPO_DIR / "requirements-colab.txt"
    if req.exists():
        run(f"pip install -q -r {req}")
    else:
        print("WARNING: requirements-colab.txt not found; installing minimal dependencies.")
        run("pip install -q numpy==2.0.2 SimpleITK pydicom nibabel scipy scikit-image matplotlib pandas requests gdown nnunetv2")


def configure_paths():
    src = REPO_DIR / "src"
    if src.exists():
        sys.path.insert(0, str(src))
        print("Added to sys.path:", src)
    else:
        print("WARNING: src directory not found:", src)


def configure_nnunet():
    os.environ["nnUNet_raw"] = "/content/nnUNet_raw"
    os.environ["nnUNet_preprocessed"] = "/content/nnUNet_preprocessed"
    os.environ["nnUNet_results"] = str(NNUNET_RESULTS_DIR)

    for d in [
        os.environ["nnUNet_raw"],
        os.environ["nnUNet_preprocessed"],
        os.environ["nnUNet_results"],
    ]:
        Path(d).mkdir(parents=True, exist_ok=True)

    print("nnU-Net environment:")
    print("  nnUNet_raw:", os.environ["nnUNet_raw"])
    print("  nnUNet_preprocessed:", os.environ["nnUNet_preprocessed"])
    print("  nnUNet_results:", os.environ["nnUNet_results"])


def check_drive_files():
    print("\\nDrive checks:")

    if FULL_DICOM_ZIP.exists():
        print("  Found:", FULL_DICOM_ZIP)
    else:
        print("  WARNING missing:", FULL_DICOM_ZIP)

    if NNUNET_RESULTS_ZIP.exists():
        print("  Found model ZIP:", NNUNET_RESULTS_ZIP)
    else:
        print("  Model ZIP not found at:", NNUNET_RESULTS_ZIP)
        print("  This is OK if /content/nnUNet_results is already populated.")


def maybe_extract_model_zip():
    """
    Extract model ZIP into /content/nnUNet_results if the folder is empty.
    Keeps this conservative: it only extracts if the zip exists and no Dataset001_CCTA_DHM is present.
    """
    target = NNUNET_RESULTS_DIR / "Dataset001_CCTA_DHM"

    if target.exists():
        print("  nnU-Net model already present:", target)
        return

    if not NNUNET_RESULTS_ZIP.exists():
        print("  Skipping model extraction.")
        return

    print("  Extracting model ZIP...")
    import zipfile
    with zipfile.ZipFile(NNUNET_RESULTS_ZIP) as z:
        z.extractall(NNUNET_RESULTS_DIR)

    print("  Extracted to:", NNUNET_RESULTS_DIR)


def main():
    print("OpenPlaque setup starting...\\n")

    install_requirements()
    configure_paths()
    configure_nnunet()
    check_drive_files()
    maybe_extract_model_zip()

    print("\\nOpenPlaque Ready")


main()
