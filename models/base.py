import torch
import torch.nn as nn

from models.stock_attentioner import StockAttentioner

class BaseModel(nn.Module):
    def __init__(self, model, preprocesser_model: nn.Module=None):
        super(BaseModel, self).__init__()
        self.model = model
        
        if preprocesser_model:
            self.preprocesser_model = preprocesser_model
            
            for param in self.preprocesser_model.parameters():
                param.requires_grad = False
            # （可選）切到 eval 模式以關閉 dropout / batchnorm 更新
            self.preprocesser_model.eval()
            
            self.linear = None
        else:
            self.preprocesser_model = None
    
    # TODO: S3E
    def preprocess_and_combine(self, x):
        preprocessed_x = self.preprocesser_model(x)
        raise NotImplementedError("Preprocessing and combining logic is not implemented.")
    
    def forward(self, x, x_broker=None, x_general=None):
        if self.preprocesser_model:
            x = self.preprocess_and_combine(x)
            
        output = self.model(x, x_broker, x_general)
        return output
            
        

    