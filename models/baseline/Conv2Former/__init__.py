from .conv2former import conv2former_t, conv2former_s, conv2former_b, conv2former_b_22k, conv2former_l, conv2former_n
from ...BaseModel import BaseModel


class Conv2Former(BaseModel):
    def __init__(self, variant='t', kernel_size=11, in_chn=2, num_classes=11):
        super().__init__()
        if variant == 't':
            self.model = conv2former_t(kernel_size=kernel_size, in_chn=in_chn, num_classes=num_classes)
        elif variant == 's':
            self.model = conv2former_s(kernel_size=kernel_size, in_chn=in_chn, num_classes=num_classes)
        elif variant == 'b':
            self.model = conv2former_b(kernel_size=kernel_size, in_chn=in_chn, num_classes=num_classes)
        elif variant == 'b_22k':
            self.model = conv2former_b_22k(kernel_size=kernel_size, in_chn=in_chn, num_classes=num_classes)
        elif variant == 'l':
            self.model = conv2former_l(kernel_size=kernel_size, in_chn=in_chn, num_classes=num_classes)
        elif variant == 'n':
            self.model = conv2former_n(kernel_size=kernel_size, in_chn=in_chn, num_classes=num_classes)
        else:
            raise ValueError("variant must be one of ['t','s','b','b_22k','l','n']")

    def forward_features(self, x):
        x = x.permute(0, 2, 1)  # B L C -> B C L
        return self.model.forward_features(x)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # B L C -> B C L

        return self.model(x)
