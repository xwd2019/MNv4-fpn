import torch.nn as nn
from .mobilenetv4 import MN4
from .block import C2f, Conv, Concat, LastConv


class Seg_no_ffm(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.mn4 = MN4()

        self.C2f_5 = C2f(416, 256, 1)
        self.C2f_6 = C2f(336, 128, 1)
        self.C2f_7 = C2f(384, 256, 1)
        self.C2f_8 = C2f(512, 512, 1)

        self.Conv_6 = Conv(128, 128, 3, 2)
        self.Conv_7 = Conv(256, 256, 3, 2)

        self.Concat = Concat(1)
        self.upSample_Bilinear = nn.Upsample(scale_factor=2,
                                    mode="bilinear",
                                    align_corners=True)
        self.upSample_PixelShuffle = nn.PixelShuffle(4)        
        self.lastConv = LastConv(in_channels=32,num_classes=num_classes)

    def forward(self, x):
        # backbone
        seg_out = self.mn4(x)
        seg_out1 = seg_out[0]  # torch.Size([N, 80, 60, 80])
        seg_out2 = seg_out[1]  # torch.Size([N, 160, 30, 40])
        seg_out3 = seg_out[2]  # torch.Size([N, 256, 15, 20])

        # feature pyramid network
        seg_out2 = self.Concat([seg_out2, self.upSample_Bilinear(seg_out3)])  # torch.Size([N, 416, 30, 40])
        seg_out2 = self.C2f_5(seg_out2)  # torch.Size([N, 256, 30, 40])
        seg_out1 = self.Concat([seg_out1, self.upSample_Bilinear(seg_out2)])  # torch.Size([N, 336, 60, 80])
        seg_out1 = self.C2f_6(seg_out1)  # torch.Size([N, 128, 60, 80])
        seg_out2 = self.Concat([self.Conv_6(seg_out1), seg_out2])  # torch.Size([N, 384, 30, 40])
        seg_out2 = self.C2f_7(seg_out2)  # torch.Size([N, 256, 30, 40])
        seg_out3 = self.Concat([self.Conv_7(seg_out2), seg_out3])  # torch.Size([N, 512, 15, 20])
        seg_out3 = self.C2f_8(seg_out3)  # torch.Size([N, 512, 15, 20])
        
        seg_out = self.upSample_PixelShuffle(seg_out3)
        seg_out = self.lastConv(seg_out)
        seg_out = self.upSample_Bilinear(self.upSample_Bilinear(self.upSample_Bilinear(seg_out)))              

        return seg_out
