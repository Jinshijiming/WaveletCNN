import hydra
from omegaconf import DictConfig

from utils.pytools import *
from classfierTask import ClassfierTask

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    cfg = init_conf(cfg)
    classfilerTask = ClassfierTask(cfg)
    classfilerTask.run()


if __name__ == '__main__':
    main()