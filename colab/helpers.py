import os, zipfile
import SimpleITK as sitk
import matplotlib.pyplot as plt

def find_file(filename, root="/content/drive/MyDrive"):
    for d,_,f in os.walk(root):
        if filename in f:
            return os.path.join(d, filename)
    raise FileNotFoundError(filename)

def unzip_if_needed(zip_path, extract_root="/content/openplaque"):
    if not os.path.exists(extract_root):
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_root)
    return extract_root

def find_dicom_dir(root):
    best=None; nbest=0
    for d,_,f in os.walk(root):
        if len(f)>nbest:
            best=d; nbest=len(f)
    return best

def load_cta(zip_name="Series7_BestDiast.zip"):
    zp=find_file(zip_name)
    ex=unzip_if_needed(zp)
    dd=find_dicom_dir(ex)
    r=sitk.ImageSeriesReader()
    sid=r.GetGDCMSeriesIDs(dd)[0]
    files=r.GetGDCMSeriesFileNames(dd,sid)
    r.SetFileNames(files)
    image=r.Execute()
    volume=sitk.GetArrayFromImage(image)
    print("Shape:",volume.shape)
    print("Spacing:",image.GetSpacing())
    print("HU:",volume.min(),volume.max())
    return image,volume,files

def show_slices(volume):
    zs=[int(volume.shape[0]*p) for p in (0.2,0.3,0.4,0.5,0.6,0.7)]
    plt.figure(figsize=(10,6))
    for i,z in enumerate(zs):
        plt.subplot(2,3,i+1)
        plt.imshow(volume[z], cmap="gray", vmin=-200, vmax=800)
        plt.axis("off")
        plt.title(f"z={z}")
    plt.tight_layout()
    plt.show()
