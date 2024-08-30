import SimpleITK as sitk

def simpleitk_loader(path):
    sitk_header = sitk.ReadImage(path)
    sitk_image = sitk.GetArrayFromImage(sitk_header)
    return sitk_image
