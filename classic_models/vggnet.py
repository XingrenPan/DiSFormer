import torch.nn as nn
import torch
import os
# official pretrain weights
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}
 
class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=False):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.features(x)
        # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)
        # N x 512*7*7
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_features(cfg: list):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)

# vgg_tiny(VGG11), vgg_small(VGG13), vgg(VGG16), vgg_big(VGG19)
cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],   
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(num_classes): 
    cfg = cfgs["vgg11"]
    model = VGG(make_features(cfg), num_classes=num_classes)
    return model

def vgg13(num_classes):  
    cfg = cfgs["vgg13"]
    model = VGG(make_features(cfg), num_classes=num_classes)
    return model

def vgg16(num_classes, pretrained=False):
    cfg = cfgs["vgg16"]
    model = VGG(make_features(cfg), num_classes=num_classes)

    if pretrained:
        weight_path = os.path.join("E:/A_PhD_AI_Project/CAIE/code/Kansformer/model_pth", "vgg16-official.pth")
        state_dict = torch.load(weight_path, map_location="cpu")

        # 删除全连接层的最后一层（适配非 ImageNet 的分类数）
        state_dict.pop("classifier.6.weight", None)
        state_dict.pop("classifier.6.bias", None)

        model.load_state_dict(state_dict, strict=False)
        print("✅ Loaded pretrained VGG16 weights (except classifier.6)")

    return model
def vgg19(num_classes):  
    cfg = cfgs['vgg19']
    model = VGG(make_features(cfg), num_classes=num_classes)
    return model