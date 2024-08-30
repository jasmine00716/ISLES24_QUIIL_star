import torch
import argparse
import random
import numpy as np
import os
import datetime

from models.maktry_model_structure import UNetWithConcat
from utils.common.hyperparameters import optimizer_fc
from utils.common.isles_data_loader_maktry import brain_dataset_preparation, dataloading_maktry
from utils.common.train_val_test_loop_maktry import train_model, val_cal

seed = 42
random.seed(seed)
np.random.seed(seed)
random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def isles24_maktry_loop(args):
    device = (f"cuda:{args.cuda_num}" if torch.cuda.is_available() else "cpu")

    model = UNetWithConcat(args.in_channel,args.out_channel, modeltype=args.model, device=device)
    pretrained_weight = torch.load('resources/weights/final_weight_2.pt', map_location=device)
    model = pretrained_weights_check(model, pretrained_weight)
    seg_loss_fn, optimizer, scheduler = optimizer_fc(model, args.init_lr)

    traindata_nii_df = brain_dataset_preparation('utils/sample_data/train')
    valdata_nii_df = brain_dataset_preparation('utils/sample_data/val')

    train_dataloader = dataloading_maktry(traindata_nii_df, shuffle=True, batch_size=args.batch_size, modality=args.modality, is_augmented=True)
    val_dataloader = dataloading_maktry(valdata_nii_df, shuffle=True, batch_size=args.batch_size, modality=args.modality, is_augmented=True)

    num_epochs = args.epochs

    month_day = datetime.datetime.today().strftime("%m%d")

    experiment_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_train_dir = f'Best_Model/{args.model}/{month_day}/{args.modality}/{experiment_time}/train'
    if not os.path.exists(output_train_dir):
        os.makedirs(output_train_dir)

    output_val_dir = f'Best_Model/{args.model}/{month_day}/{args.modality}/{experiment_time}/val'
    if not os.path.exists(output_val_dir):
        os.makedirs(output_val_dir)

    best_val_loss, best_train_loss = 100000000, 100000000
    for currentepoch in range(num_epochs):
        print(f"Epoch {currentepoch + 1} \n ------------------------")
        train_loss_perepoch = train_model(train_dataloader, model, seg_loss_fn, optimizer, device)
        val_f1, val_lesion_cnt, val_dice, val_volume, val_loss = val_cal(currentepoch, val_dataloader, model, seg_loss_fn, 0.3, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

            torch.save(best_model.state_dict(), f"{output_val_dir}/ISLES24_{args.cuda_num}_{args.model}_{args.init_lr}_{args.epochs}_{currentepoch}.pt")

        if train_loss_perepoch < best_train_loss:
            best_train_loss = train_loss_perepoch
            best_model = model

            torch.save(best_model.state_dict(), f"{output_train_dir}/ISLES24_{args.cuda_num}_{args.model}_{args.init_lr}_{args.epochs}_{currentepoch}.pt")

        scheduler.step()

def pretrained_weights_check(new_model, pretrained_weights):
    model_state_dict = new_model.state_dict()
    for name, param in pretrained_weights.items():
        if name in model_state_dict:
            if param.shape == model_state_dict[name].shape:
                model_state_dict[name].copy_(param)
            else:
                print(f"Skipping {name} due to shape mismatch: {param.shape} vs {model_state_dict[name].shape}")

    new_model.load_state_dict(model_state_dict)
    return new_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISLES24 model training script")
    parser.add_argument('--cuda_num', type=int, default=0)
    parser.add_argument('--init_lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--model', type=str, default='transformer')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--in_channel', type=int, default=4)
    parser.add_argument('--out_channel', type=int, default=1)
    parser.add_argument('--modality', type=str, default='tmax')

    args = parser.parse_args()

    isles24_maktry_loop(args)

# python main.py --cuda_num 0 --init_lr 0.0001 --epochs 100 --model 'cnn' --batch_size 1 --in_channel 3 --modality 'tmaxoff'
# python main.py --cuda_num 1 --init_lr 0.0001 --epochs 100 --model 'cnn' --batch_size 1 --in_channel 4 --modality 'tmax'

# python main.py --cuda_num 2 --init_lr 0.0001 --epochs 100 --model 'transformer' --batch_size 1 --in_channel 3 --modality 'tmaxoff'
# python main.py --cuda_num 3 --init_lr 0.0001 --epochs 100 --model 'transformer' --batch_size 1 --in_channel 4 --modality 'tmax'

# python main.py --cuda_num 4 --init_lr 0.0001 --epochs 100 --model 'cnn' --batch_size 1 --in_channel 4 --modality 'tmax'