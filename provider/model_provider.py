from models import Model_Dict


def modelloader(cfg, task='cls'):
    if cfg.model['name'] == 'SigNet':
        model = Model_Dict[cfg.model['name']]()
    else:
        config = {**cfg.model, **cfg.dataset}
        model = Model_Dict[cfg.model['name']].from_config(config)

    return model
