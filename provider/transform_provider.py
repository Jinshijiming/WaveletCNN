from dataset import iqtransforms as T


def transform_provider(model_name):
    '''
    每个模型需要的变换增强不一样，需要特别处理。
    可插拔设计，没有提供变换增强也不影响。
    #TODO: 每次添加需要变换的模型，都需要维护这个函数，而且与模型耦合度高，是否可在模型文件中维护？

    :param model_name: 模型名称
    :return: 单个变换，或Compose变换
    '''
    if model_name in ['ULCNN','IQGCA','STF_GCN']:
        return T.GivenTransformation()
    # elif model_name in ['WaveletCNN']:
    #     # return T.AddNoise()
    #     return T.GivenTransformation()
    else:
        return None
