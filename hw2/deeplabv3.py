import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import build_ResNet
from decoder import build_decoder
from aspp import build_aspp

class DeepLabv3(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=9):
        super(DeepLabv3, self).__init__()

        self.backbone = build_ResNet(output_stride, nn.BatchNorm2d, pretrained=True)
        self.aspp = build_aspp(2048, output_stride)
        self.decoder = build_decoder(num_classes)

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x



# if __name__ == "__main__":
#     model = DeepLab(backbone='mobilenet', output_stride=16)
#     model.eval()
#     input = torch.rand(1, 3, 513, 513)
#     output = model(input)
#     print(output.size())