import torch.nn as nn
from .mobilenetv4 import MN4
from .block import Decoder, ASPP, LastConv


class Seg_no_fpn(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.mn4 = MN4()

        self.upSample = nn.Upsample(scale_factor=2,
                                    mode="bilinear",
                                    align_corners=True)

        self.lastConv1 = LastConv(in_channels=80, num_classes=128)
        self.lastConv2 = LastConv(in_channels=160, num_classes=256)
        self.lastConv3 = LastConv(in_channels=256, num_classes=512)

        self.aspp_1 = ASPP(inplanes=512)
        self.aspp_2 = ASPP(inplanes=256)
        self.decoder_1 = Decoder(low_level_channels=256, output_channels=256)
        self.decoder_2 = Decoder(output_channels=num_classes)

    def forward(self, x):
        # backbone
        seg_out = self.mn4(x)
        seg_out1 = seg_out[0]  # torch.Size([N, 80, 60, 80])
        seg_out2 = seg_out[1]  # torch.Size([N, 160, 30, 40])
        seg_out3 = seg_out[2]  # torch.Size([N, 256, 15, 20])

        seg_out1 = self.lastConv1(seg_out1)
        seg_out2 = self.lastConv2(seg_out2)
        seg_out3 = self.lastConv3(seg_out3)

        # feature fusion module
        seg_out3 = self.aspp_1(seg_out3)  # torch.Size([N, 256, 15, 20])
        seg_out2 = self.decoder_1(x=seg_out3, low_level_feat=seg_out2)  # torch.Size([N, 256, 30, 40])
        seg_out2 = self.aspp_2(seg_out2)  # torch.Size([N, 256, 30, 40])
        seg_out = self.decoder_2(x=seg_out2, low_level_feat=seg_out1)  # torch.Size([1, 6, 60, 80])
        seg_out = self.upSample(self.upSample(self.upSample(seg_out)))  # torch.Size([1, 6, 480, 640])

        return seg_out
