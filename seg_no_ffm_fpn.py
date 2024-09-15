import torch.nn as nn
from .mobilenetv4 import MN4
from .block import LastConv


class Seg_no_ffm_fpn(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.mn4 = MN4()
        self.lastConv = LastConv(in_channels=16, num_classes=num_classes)
        self.upSample_PixelShuffle = nn.PixelShuffle(4)
        self.upSample_Bilinear = nn.Upsample(scale_factor=2,
                                             mode="bilinear",
                                             align_corners=True)

    def forward(self, x):
        # backbone
        seg_out3 = self.mn4(x)[2]  # torch.Size([N, 256, 15, 20])
        seg_out = self.upSample_PixelShuffle(seg_out3)
        seg_out = self.lastConv(seg_out)
        seg_out = self.upSample_Bilinear(self.upSample_Bilinear(self.upSample_Bilinear(seg_out)))

        return seg_out
