from ctypes.macholib.framework import framework_info

from .avgnet import Net as AvgNet
from .ConvMixer import ConvMixer
from .FEA_T import Net as FEA_T
from .lstm import lstm2 as LSTM
# # from .mcldnn import MCLDNN
from .mcldnn import Mcldnn as MCLDNN
from .SigNet import ResNet50 as SigNet
from .CTDNN import CTDNN
from .SigFormer import SigFormer
from .MCFormer import MCformer
from .WAN import WAN
from .ASN import AttentiveSiameseNetwork as ASN
from .ResNet import ResNet
from .STF_GCN import Net as STF_GCN
from .IQFormer import IQFormer
from .AWN import AWN
from .IQGCA import Net as IQGCA
from .Conv2Former import Conv2Former
from .AWN_official import AWN as AWN_official
from .ULCNN import ULCNN
from .ArF_AMC import ArFAMCNet as ArF_AMC
from .DNNTF import DNNTF
from .CNN1D import ResNet1D as CNN1D
from .ALWNN import ALWNN
from .AMC_Net import AMC_Net