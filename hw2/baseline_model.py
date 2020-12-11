import torch
import torch.nn as nn
import torchvision.models as models
from networks import LinearBlock, Conv2dBlock, ConvTranspose2dBlock
 


class Net(nn.Module):

    def __init__(self, args):
        super(Net, self).__init__()  
        layers = []
        self.n_layers = 5
        self.bottle_dim = 8
        self.norm_fn = args.norm_fn
        self.acti_fn = args.acti_fn
        model_ft = models.resnet18(pretrained=True)
        self.resnet_feature = torch.nn.Sequential(*list(model_ft.children())[:8])

        n_in = 512
        n_out = 256
        num_classes = 9
        for i in range(self.n_layers):
            layers += [ConvTranspose2dBlock(
                n_in, n_out,  stride=2, padding=1, norm_fn= self.norm_fn, acti_fn= self.acti_fn
            )]
            n_in = n_out
            n_out = int(n_out/ 2)

        self.conv_trans_layers = nn.Sequential(*layers)
        self.conv_layer = Conv2dBlock(16, num_classes, stride=1, padding=0, norm_fn= 'none', acti_fn= 'none')
        # self.softmax = nn.Softmax(dim = 1)


    def forward(self, img):
        img = self.resnet_feature(img)
        img = self.conv_trans_layers(img)
        img = self.conv_layer(img)
        # img = self.softmax(img)

        return img
