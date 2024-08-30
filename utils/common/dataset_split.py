import numpy as np
import pandas as pd
from skimage import measure
import nibabel as nib


root_dir = 'file_dir_csvs/whole_data_nii_path.csv'

whole_dataset_df = pd.read_csv(root_dir)

total_lesion_volume = []
whole_cnt = 0
cnt_label = []
lesion_cnt = []
small, med, big = [], [], []
for idx in range(len(whole_dataset_df)):
    gt = nib.load(whole_dataset_df.iloc[idx, 11]).get_fdata()

    gt[gt > 0] = 1

    new_label = measure.label(gt)
    cnt_label.append(new_label.max())

    whole_cnt = np.sum(gt > 0.5)
    lesion_cnt.append(whole_cnt)

    if new_label.max() == 0:
        ratio = whole_cnt / 1
    else:
        ratio = whole_cnt / new_label.max()

    if ratio < 1200:
        small.append(whole_dataset_df.iloc[idx, 11])
    elif 1200 <= ratio < 7000:
        med.append(whole_dataset_df.iloc[idx, 11])
    else:
        big.append(whole_dataset_df.iloc[idx, 11])

small_df = pd.DataFrame(small)
med_df = pd.DataFrame(med)
big_df = pd.DataFrame(big)

train_small = small_df.sample(frac=0.8, random_state=42)
remain_small = small_df.drop(train_small.index)
val_small = remain_small.sample(frac=0.1 / (0.1 + 0.1), random_state=42)
test_small = remain_small.drop(val_small.index)

train_med = med_df.sample(frac=0.8, random_state=42)
remain_med = med_df.drop(train_med.index)
val_med = remain_med.sample(frac=0.1 / (0.1 + 0.1), random_state=42)
test_med = remain_med.drop(val_med.index)

train_big = big_df.sample(frac=0.8, random_state=42)
remain_big = big_df.drop(train_big.index)
val_big = remain_big.sample(frac=0.1 / (0.1 + 0.1), random_state=42)
test_big = remain_big.drop(val_big.index)

trainset = pd.concat([train_small, train_med, train_big])
valset = pd.concat([val_small, val_med, val_big])
testset = pd.concat([test_small, test_med, test_big])

trainset.to_csv("file_dir_csvs/isles24_trainset_split(0802).csv", index=False)
valset.to_csv("file_dir_csvs/isles24_valset_split(0802).csv", index=False)
testset.to_csv("file_dir_csvs/isles24_testset_split(0802).csv", index=False)