import os
import zipfile
import SimpleITK as sitk


def find_file(filename, root="/content/drive/MyDrive"):
    for dirpath, _, files in os.walk(root):
        if filename in files:
            return os.path.join(dirpath, filename)
    raise FileNotFoundError(f"Could not find {filename} under {root}")


def unzip_if_needed(zip_path, extract_root="/content/openplaque"):
    if not os.path.exists(extract_root):
        os.makedirs(extract_root, exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_root)
    return extract_root


def find_dicom_dir(root):
    best_dir = None
    best_count = 0

    for dirpath, _, files in os.walk(root):
        count = len(files)
        if count > best_count:
            best_count = count
            best_dir = dirpath

    if best_dir is None:
        raise FileNotFoundError("No DICOM directory found.")

    return best_dir, best_count


def load_dicom_series(dicom_dir):
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)

    if not series_ids:
        raise RuntimeError(f"No DICOM series found in {dicom_dir}")

    files = reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0])
    reader.SetFileNames(files)
    image = reader.Execute()
    volume = sitk.GetArrayFromImage(image)

    return image, volume, files


def load_cta(zip_name="Series7_BestDiast.zip",
             drive_root="/content/drive/MyDrive",
             extract_root="/content/openplaque"):
    zip_path = find_file(zip_name, drive_root)
    extract_root = unzip_if_needed(zip_path, extract_root)
    dicom_dir, n_files = find_dicom_dir(extract_root)
    image, volume, files = load_dicom_series(dicom_dir)

    print("ZIP:", zip_path)
    print("DICOM dir:", dicom_dir)
    print("Files:", len(files))
    print("Shape:", volume.shape)
    print("Spacing:", image.GetSpacing())
    print("HU:", volume.min(), volume.max())

    return image, volume, files
