from pathlib import Path
from omegaconf import OmegaConf
import torch
from provider import modelloader


class InferenceClass():
    '''
    推理类，用于加载模型和规划推理步骤
    '''
    def __init__(self,dir,device,cfg_fn='config.yaml',pth_fn='clsBestModel.pth',class_list=None):
        '''
        __init__ 的 Docstring
        
        :param self: 说明
        :param dir: cfg和pth所在目录
        :param device: 说明
        :param cfg_fn: cfg的文件名
        :param pth_fn: pth的文件名
        '''
        #模型加载
        path=Path(dir)
        cfg = OmegaConf.load(path / cfg_fn)
        model_pth=str(path / pth_fn)
        self.device=device
        
        self.model=modelloader(cfg).to(self.device)
        state=torch.load(model_pth,map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        torch.no_grad()
        #类别列表
        
        self.class_list=class_list
            
    def get_model(self):
        return self.model
    def infer_step(self,input,output_pred=False):
        """
        
        :param input: B L C
        :param output_pred: 是否输出预测类别
        """
        output=self.model.infer_step(input)
        
        if not output_pred:
            return output
        else:
            pred=output['pred']

            results = [
                f"Known class: {self.class_list[p]}"
                for p in pred
            ]

            return results

if __name__ == "__main__":
    import numpy as np
    model_dir='/public/bsh/Signal/TrainSystem/outputs/CAC_CNN1D/RML128__allSNR_1'
    device='cuda:5'
    
    infer_model=InferenceClass(model_dir,device)
    
    # 生成随机输入数据
    input=torch.randn(2,128,2)
    
    result=infer_model.infer(input,output_pred=True)
    print(result)