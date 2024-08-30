import SimpleITK as sitk
import os
import numpy as np
from skimage.transform import resize
import pandas as pd

def simpleitk_loader(path):
    sitk_header = sitk.ReadImage(path)
    sitk_image = sitk.GetArrayFromImage(sitk_header)
    return sitk_image

if __name__ == "__main__":
    gt = simpleitk_loader("utils/sample_data/train/sub-stroke0003_ses-02_lesion-msk.nii.gz")
    adc = simpleitk_loader('utils/sample_data/train/sub-stroke0003_ses-02_adc.nii.gz')
    dwi = simpleitk_loader('utils/sample_data/train/sub-stroke0003_ses-02_dwi.nii.gz')
    cta = simpleitk_loader('utils/sample_data/train/sub-stroke0003_ses-01_cta.nii.gz')
    ncct = simpleitk_loader('utils/sample_data/train/sub-stroke0003_ses-01_ncct.nii.gz')
    ctp = simpleitk_loader('utils/sample_data/train/sub-stroke0003_ses-01_ctp.nii.gz')
    tmax = simpleitk_loader('utils/sample_data/train/sub-stroke0003_ses-01_space-ncct_tmax.nii.gz')

    patient_path = 'preprocessed_npy'
    if not os.path.exists(patient_path):
        os.makedirs(patient_path)

    adc = resize(adc, (40, 224, 224))
    dwi = resize(dwi, (40, 224, 224))
    gt = resize(gt, (40, 224, 224))
    cta = resize(cta, (40, 224, 224))
    ncct = resize(ncct, (40, 224, 224))
    tmax = resize(ncct, (40, 224, 224))
    ctp = resize(ctp, (55, 40, 224, 224))

    np.save("preprocessed_npy/gt.npy", gt)
    np.save("preprocessed_npy/dwi.npy", dwi)
    np.save("preprocessed_npy/adc.npy", adc)
    np.save("preprocessed_npy/ncct.npy", ncct)
    np.save("preprocessed_npy/cta.npy", cta)
    np.save("preprocessed_npy/tmax.npy", tmax)
    np.save("preprocessed_npy/ctp.npy", ctp)

    adc_path, dwi_path, gt_path, cta_path, ncct_path, ctp_path, tmax_path = [], [], [], [], [], [], []
    gt_path.append("preprocessed_npy/gt.npy")
    dwi_path.append("preprocessed_npy/dwi.npy")
    adc_path.append("preprocessed_npy/adc.npy")
    ncct_path.append("preprocessed_npy/ncct.npy")
    cta_path.append("preprocessed_npy/cta.npy")
    ctp_path.append("preprocessed_npy/ctp.npy")
    tmax_path.append("preprocessed_npy/tmax.npy")

    file_dir_pd = pd.DataFrame({"GT" : gt_path,
                                "DWI" : dwi_path,
                                "ADC" : adc_path,
                                "NCCT" : ncct_path,
                                "CTA" : cta_path,
                                "CTP" : ctp_path,
                                "TMAX" : tmax_path
                                })