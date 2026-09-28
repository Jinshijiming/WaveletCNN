from .pytools import *
from omegaconf import OmegaConf


OmegaConf.register_new_resolver('version', dir_increment, use_cache=True)
OmegaConf.register_new_resolver('eval', eval)
