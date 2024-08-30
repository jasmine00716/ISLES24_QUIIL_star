import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from skimage.transform import resize
from scipy.ndimage import rotate
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import SimpleITK as sitk
import random
import torchio as tio
from utils.common.nii_to_npy import simpleitk_loader
import os
import glob

def __itensity_normalize_one_volume__(volume):
    """
    normalize the itensity of an nd volume based on the mean and std of nonzeor region
    inputs:
        volume: the input nd volume
    outputs:
        out: the normalized nd volume
    """
    pixels = volume[volume > 0]
    mean = pixels.mean()
    std = pixels.std()
    out = (volume - mean) / std
    out_random = np.random.normal(0, 1, size=volume.shape)
    out[volume == 0] = out_random[volume == 0]
    return out


def min_max_norm(input_modality):
    return (input_modality - input_modality.min()) / (input_modality.max() - input_modality.min())


def brain_dataset_preparation(path):
    gt = simpleitk_loader(glob.glob(f"{path}/*_lesion-msk.nii.gz")[0])
    adc = simpleitk_loader(glob.glob(f"{path}/*_adc.nii.gz")[0])
    dwi = simpleitk_loader(glob.glob(f"{path}/*_dwi.nii.gz")[0])
    cta = simpleitk_loader(glob.glob(f"{path}/*_space-ncct_cta.nii.gz")[0])
    ncct = simpleitk_loader(glob.glob(f"{path}/*_ncct.nii.gz")[0])
    ctp = simpleitk_loader(glob.glob(f"{path}/*_space-ncct_ctp.nii.gz")[0])
    tmax = simpleitk_loader(glob.glob(f"{path}/*_space-ncct_tmax.nii.gz")[0])

    patient_path = f'{path}/preprocessed_npy'
    if not os.path.exists(patient_path):
        os.makedirs(patient_path)

    nii_gt = []
    nii_gt.append(glob.glob(f"{path}/*_lesion-msk.nii.gz")[0])

    adc = resize(adc, (40, 224, 224))
    dwi = resize(dwi, (40, 224, 224))
    gt = resize(gt, (40, 224, 224))
    cta = resize(cta, (40, 224, 224))
    ncct = resize(ncct, (40, 224, 224))
    tmax = resize(tmax, (40, 224, 224))
    ctp = resize(ctp, (55, 40, 224, 224))

    np.save(f"{patient_path}/gt.npy", gt)
    np.save(f"{patient_path}/dwi.npy", dwi)
    np.save(f"{patient_path}/adc.npy", adc)
    np.save(f"{patient_path}/ncct.npy", ncct)
    np.save(f"{patient_path}/cta.npy", cta)
    np.save(f"{patient_path}/tmax.npy", tmax)
    np.save(f"{patient_path}/ctp.npy", ctp)

    adc, gt_npy, ncct, cta, ctp, tmax = [], [], [], [], [], []
    adc.append(f"{patient_path}/adc.npy")
    gt_npy.append(f"{patient_path}/gt.npy")
    ncct.append(f"{patient_path}/ncct.npy")
    cta.append(f"{patient_path}/cta.npy")
    ctp.append(f"{patient_path}/ctp.npy")
    tmax.append(f"{patient_path}/tmax.npy")

    df = pd.DataFrame({"nii_gt": nii_gt,
                       "adc": adc,
                       "ncct": ncct,
                       "cta": cta,
                       "ctp": ctp,
                       "gt_npy": gt_npy,
                       "tmax": tmax})

    return df


def brain_dataset_preparation_Maktry(path): # dataframe 으로 file path 뽑기
    dataset_df = pd.read_csv(path)

    adc, gt_npy, ncct, cta, ctp, nii_gt, tmax = [], [], [], [], [], [], []
    for i in range(len(dataset_df)):
        nii_gt.append(dataset_df.iloc[i][0])
        batch_num = dataset_df.iloc[i][0].split('/')[3]
        patient_num = dataset_df.iloc[i][0].split('/')[5]

        base_dir = "/data2/braindata/preprocessed_npy_0827_maktry"
        adc.append(f"{base_dir}/{batch_num}_npy/{patient_num}/adc.npy")
        ncct.append(f"{base_dir}/{batch_num}_npy/{patient_num}/ncct.npy")
        cta.append(f"{base_dir}/{batch_num}_npy/{patient_num}/cta.npy")
        ctp.append(f"{base_dir}/{batch_num}_npy/{patient_num}/ctp.npy")
        tmax.append(f"{base_dir}/{batch_num}_npy/{patient_num}/tmax.npy")
        gt_npy.append(f"{base_dir}/{batch_num}_npy/{patient_num}/gt.npy")

    df = pd.DataFrame({"nii_gt": nii_gt,
                       "adc": adc,
                       "ncct": ncct,
                       "cta": cta,
                       "ctp": ctp,
                       "gt_npy": gt_npy,
                       "tmax": tmax})

    return df


class BrainDWIDataset_MakTry(Dataset):
    def __init__(self, df, modality, is_augmented):
        self.df = df
        self.is_augmented = is_augmented
        self.modality = modality
        self.rotation_transform = tio.transforms.RandomAffine(
            scales=(1, 1, 1),
            degrees=(45, 0, 0),
            translation=(0, 0, 0),
            p=1
        )
        self.flip_transform = tio.transforms.RandomFlip(axes=1, flip_probability=1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        voxel_volume = np.prod(nib.load(self.df.iloc[idx, 0]).header.get_zooms()) / 1000  # Get voxel volume
        ncct = torch.Tensor(min_max_norm(np.load(self.df.iloc[idx, 2]).astype(np.float64)))
        cta = torch.Tensor(min_max_norm(np.load(self.df.iloc[idx, 3]).astype(np.float64)))
        ctp = torch.Tensor(min_max_norm(np.load(self.df.iloc[idx, 4]).astype(np.float64)))

        tmax = np.load(self.df.iloc[idx, 6]).astype(np.float64)

        tmax_cutoff = tmax > 9
        tmax_final = torch.Tensor(tmax_cutoff)

        gt_npy = torch.Tensor(np.load(self.df.iloc[idx, 5]).astype(np.float64))

        rotate_p, flip_p = random.random(), random.random()
        if self.is_augmented:
            if rotate_p < 0.5:
                ncct = self.rotation_transform(ncct.unsqueeze(0)).squeeze(0)
                cta = self.rotation_transform(cta.unsqueeze(0)).squeeze(0)
                tmax_final = self.rotation_transform(tmax_final.unsqueeze(0)).squeeze(0)
                ctp = self.rotation_transform(ctp)
                gt_npy = self.rotation_transform(gt_npy.unsqueeze(0)).squeeze(0)

            if flip_p < 0.5:
                ncct = self.flip_transform(ncct.unsqueeze(0)).squeeze(0)
                cta = self.flip_transform(cta.unsqueeze(0)).squeeze(0)
                tmax_final = self.flip_transform(tmax_final.unsqueeze(0)).squeeze(0)
                ctp = self.flip_transform(ctp)
                gt_npy = self.flip_transform(gt_npy.unsqueeze(0)).squeeze(0)

        if self.modality == 'tmax':
            rest_modality = torch.stack((ncct, cta, tmax_final), dim=0)
        else:
            rest_modality = torch.stack((ncct, cta), dim=0)

        gt_npy = gt_npy.unsqueeze(0)

        return rest_modality, ctp, gt_npy, voxel_volume

def dataloading_maktry(df, shuffle, batch_size=1, modality='', is_augmented=False):
    dataset = BrainDWIDataset_MakTry(df=df, modality=modality, is_augmented=is_augmented)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader



if __name__ == '__main__':
    root_dir = "/data1/braindata/file_dir_csvs"
    sample = brain_dataset_preparation(root_dir + "/MR_trainset_split(0709).csv")
    train_dataloader = dataloading_maktry(sample, shuffle=True, batch_size=1)
