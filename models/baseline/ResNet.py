"""
"constellation diagram(CD) and ResNet"
reproduced by ASME

"""
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torch import nn
from .. import BaseModel

class ResNet(BaseModel):
    def __init__(self, layers=50, pretrained=False, num_classes=11):
        super().__init__()
        if layers == 18:
            self.model = resnet18(pretrained=pretrained, num_classes=num_classes)
        elif layers == 34:
            self.model = resnet34(pretrained=pretrained, num_classes=num_classes)
        elif layers == 50:
            self.model = resnet50(pretrained=pretrained, num_classes=num_classes)
        elif layers == 101:
            self.model = resnet101(pretrained=pretrained, num_classes=num_classes)
        elif layers == 152:
            self.model = resnet152(pretrained=pretrained, num_classes=num_classes)
        else:
            raise ValueError("Unsupported ResNet layer count: {}".format(layers))

        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        if x.ndim == 5:
            x = x.squeeze(1)
        return self.model(x)


