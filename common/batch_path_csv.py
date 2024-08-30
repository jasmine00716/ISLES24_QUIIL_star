import pandas as pd
import numpy as np
import glob
import os

root_dir = '/data1/braindata/batch_*/derivatives/*'

gt_dir = '/data1/braindata/batch_*/derivatives/*/ses-02/*'

preprocessed_cta_dir = '/data1/braindata/batch_*/derivatives/*/ses-01/*cta.nii.gz'
preprocessed_ctp_dir = '/data1/braindata/batch_*/derivatives/*/ses-01/*ctp.nii.gz'

raw_cta_dir = '/data1/braindata/batch_*/raw_data/*/ses-01/*cta.nii.gz'
raw_ctp_dir = '/data1/braindata/batch_*/raw_data/*/ses-01/*ctp.nii.gz'
raw_ncct_dir = '/data1/braindata/batch_*/raw_data/*/ses-01/*ncct.nii.gz'

mr_dwi_dir = '/data1/braindata/batch_*/raw_data/*/ses-02/*dwi.nii.gz'
mr_adc_dir = '/data1/braindata/batch_*/raw_data/*/ses-02/*adc.nii.gz'

perfusion_dir = root_dir + '/ses-01/perfusion-maps/*'

preprocessed_cta_path = glob.glob(preprocessed_cta_dir)
preprocessed_ctp_path = glob.glob(preprocessed_ctp_dir)
raw_cta_path = glob.glob(raw_cta_dir)
raw_ctp_path = glob.glob(raw_ctp_dir)
raw_ncct_path = glob.glob(raw_ncct_dir)
mr_dwi_path = glob.glob(mr_dwi_dir)
mr_adc_path = glob.glob(mr_adc_dir)
perfusion_path = glob.glob(perfusion_dir)
gt_path = glob.glob(gt_dir)

raw_cta, raw_ctp, prepro_cta, prepro_ctp, raw_ncct, mr_dwi, mr_adc, cbf, cbv, mtt, tmax, gt = [], [], [], [], [], [], [], [], [], [], [], []

whole_dataset = [preprocessed_cta_path, preprocessed_ctp_path, raw_cta_path, raw_ctp_path, raw_ncct_path, mr_dwi_path, mr_adc_path, perfusion_path, gt_path]

for i in range(len(whole_dataset)):
    if i == 0:
        for file_name in whole_dataset[i]:
            prepro_cta.append(file_name)
    elif i == 1:
        for file_name in whole_dataset[i]:
            prepro_ctp.append(file_name)
    elif i == 2:
        for file_name in whole_dataset[i]:
            raw_cta.append(file_name)
    elif i == 3:
        for file_name in whole_dataset[i]:
            raw_ctp.append(file_name)
    elif i == 4:
        for file_name in whole_dataset[i]:
            raw_ncct.append(file_name)
    elif i == 5:
        for file_name in whole_dataset[i]:
            mr_dwi.append(file_name)
    elif i == 6:
        for file_name in whole_dataset[i]:
            mr_adc.append(file_name)
    elif i == 7:
        for file_name in whole_dataset[i]:
            if 'cbf' in file_name:
                cbf.append(file_name)
            elif 'cbv' in file_name:
                cbv.append(file_name)
            elif 'mtt' in file_name:
                mtt.append(file_name)
            else:
                tmax.append(file_name)
    else:
        for file_name in whole_dataset[i]:
            gt.append(file_name)

raw_cta = sorted(raw_cta)
raw_ctp = sorted(raw_ctp)
prepro_cta = sorted(prepro_cta)
prepro_ctp = sorted(prepro_ctp)
raw_ncct = sorted(raw_ncct)
mr_dwi = sorted(mr_dwi)
mr_adc = sorted(mr_adc)
cbf = sorted(cbf)
cbv = sorted(cbv)
mtt = sorted(mtt)
tmax = sorted(tmax)
gt = sorted(gt)

ct_mri_gt = pd.DataFrame({"Raw_CTA" : raw_cta,
                          "Prepro_CTA" : prepro_cta,
                          "Raw_CTP" : raw_ctp,
                          "Prepro_CTP" : prepro_ctp,
                          "Raw_NCCT" : raw_ncct,
                          "ADC" : mr_adc,
                          "DWI" : mr_dwi,
                          "CBF" : cbf,
                          "CBV" : cbv,
                          "MTT" : mtt,
                          "TMAX" : tmax,
                          "GT" : gt})

ct_mri_gt.to_csv("file_dir_csvs/whole_data_nii_path.csv", index=False)

print()
