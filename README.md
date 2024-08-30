# ISLES24_QUiiL_star

## Instructions
1. Create directory "resources/weights"
2. Download pretrained weight from

    https://drive.google.com/file/d/1af6u3eBRlzoPA_Ycmdz8twgU_L8MS6Qd/view?usp=sharing
5. Add "final_weight_2.pt" to "resources/weights"

The structure of repository should look like this:
```bash
ISLES24_QUIIL_star/
├── Best_Model
├── file_dir_csvs/
│   └── ...
├── models/
│   ├── MoReT_3D/
│   │   └── mobilevit_v3_block.py
│   │   └── moret_3d.py
│   │   └── vit_block.py
│   └── model_structure.py
├── resources/
│   ├── weights/
│   │   └── final_weight_2.pt  # This will be downloaded through the link above.
├── utils/
│   ├── common/
│   │   └── ...
│   ├── sample_data/  # These are selected randomly for test.
│   │   └── train/ 
│   │   │   └── sub-stokre0003_ses-01_cta.nii.gz
│   │   │   └── sub-stroke0003_ses-01_ctp.nii.gz
│   │   │   └── sub-stroke0003_ses-01_ncct.nii.gz
│   │   │   └── sub-stroke0003_ses-01_space-ncct_cta.nii.gz
│   │   │   └── sub-stroke0003_ses-01_space-ncct_ctp.nii.gz
│   │   │   └── sub-stroke0003_ses-01_space-ncct_tmax.nii.gz
│   │   │   └── sub-stroke0003_ses-02_adc.nii.gz
│   │   │   └── sub-stroke0003_ses-02_dwi.nii.gz
│   │   │   └── sub-stroke0003_ses-02_lesion-msk.nii.gz
│   │   └── val/
│   │   │   └── sub-stroke0004_ses-01_ncct.nii.gz
│   │   │   └── sub-stroke0004_ses-01_space-ncct_cta.nii.gz
│   │   │   └── sub-stroke0004_ses-01_space-ncct_ctp.nii.gz
│   │   │   └── sub-stroke0004_ses-01_space-ncct_tmax.nii.gz
│   │   │   └── sub-stroke0004_ses-02_adc.nii.gz
│   │   │   └── sub-stroke0004_ses-02_dwi.nii.gz
│   │   │   └── sub-stroke0004_ses-02_lesion-msk.nii.gz
│   └── isles_eval_util.py
└── main.py
```
