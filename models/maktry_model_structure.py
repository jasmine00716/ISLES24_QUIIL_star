import torch.nn as nn
import torch
from models.MoReT_3D.moret_3d import MoReT_3D

def moret(in_channel):
    input_channel = in_channel
    network_architecture = {
            "parameters": {
                "image_size": (40, 224, 224),
                "dims": [64, 80, 96],
                "channels": [16, 16, 16, 32, 32, 32, 64, 64, 128, 128, 256, 256],
                "input_channel": input_channel,
                "kernel_size": 3,
                "patch_size": (2, 2, 2),
                "num_classes": 1,
                "expansion": 2,
                "device": torch.device(f"cuda"),
            }
        }

    params = network_architecture['parameters']
    network = MoReT_3D(**params)
    return network


class UNetWithConcat(nn.Module):
    def __init__(self, in_channel, out_channel, modeltype, device):
        super(UNetWithConcat, self).__init__()
        self.device = device

        self.seg_model = moret(in_channel).to(device)

        self.reduce_channel_sub_ctp = nn.Conv3d(in_channels=55, out_channels=1, kernel_size=1, stride=1, padding=0).to(device)

    def forward(self, rest_modality, sub_ctp):
        reduced_sub_ctp = self.reduce_channel_sub_ctp(sub_ctp.to(self.device))  # (batch_size, 1, 40, 512, 512)

        combined_input = torch.cat([rest_modality.to(self.device), reduced_sub_ctp], dim=1)  # (batch_size, 5, 40, 512, 512)

        seg_output = self.seg_model(combined_input)

        return seg_output


if __name__ == "__main__":
    model = UNetWithConcat(3, 1, modeltype = 'sumin',device ="cuda:0")

    input1 = torch.randn(1, 2, 40, 224, 224).to("cuda:0")
    input3 = torch.randn(1, 55, 40, 224, 224).to("cuda:0")

    output = model(input1, input3)
