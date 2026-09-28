from .dataset import RML_2016a, RML_2016b, RML_2018a, Sig2019_12,HKDD36,TechRec
from .constellation_dataset import RML_2016a_CD, RML_2016b_CD, RML_2018a_CD, Sig2019_CD
from .stft_dataset import RML_2016a_stft, RML_2016b_stft, RML_2018a_stft, Sig2019_stft,TechRec_stft

Data_Dict = {
    'RML128': RML_2016a,
    'RML128b': RML_2016b,
    'RML1024': RML_2018a,
    'SIG512': Sig2019_12,
    'RML128_CD': RML_2016a_CD,
    'RML128b_CD': RML_2016b_CD,
    'RML1024_CD': RML_2018a_CD,
    'SIG512_CD': Sig2019_CD,
    'RML128_STFT': RML_2016a_stft,
    'RML128b_STFT': RML_2016b_stft,
    'RML1024_STFT': RML_2018a_stft,
    'SIG512_STFT': Sig2019_stft,
    'TechRec_STFT': TechRec_stft,
    'HKDD36': HKDD36,
    'TechRec': TechRec,
}
