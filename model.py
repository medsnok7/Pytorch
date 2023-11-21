import torch 
from torch import nn 

class ChatBotModel(nn.Module):
    def __init__(self,input_features,output_features,hidden_layer=64):
        super().__init__()
        self.model=nn.Sequential(
            nn.Linear(in_features=input_features,out_features=hidden_layer),
            nn.ReLU(),
            nn.Linear(in_features=hidden_layer,out_features=hidden_layer),
            nn.ReLU(),
            nn.Linear(in_features=hidden_layer,out_features=output_features)
        )
    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.model(x)
    
