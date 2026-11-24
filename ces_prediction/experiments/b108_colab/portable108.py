"""Portable exact Ti branch and frozen numerical objective; no repo imports."""
import torch
from torch import nn
class TemperatureLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm_ti=nn.LSTM(22,160,num_layers=2,batch_first=True,dropout=.1)
        self.norm_ti=nn.LayerNorm(160)
        self.head_ti=nn.Sequential(nn.Linear(160,64),nn.GELU(),nn.Linear(64,1))
    def forward(self,x,lengths=None):
        h,_=self.lstm_ti(x[...,:22]);f=self.head_ti(self.norm_ti(h));return torch.cat([f,f*0],dim=-1)
def loss108(pred,y,mask,mu,sd):
    m=mask.to(pred.dtype);n=m.sum((0,1));has=(n>0).to(pred.dtype)
    error=torch.where(mask,pred-y,torch.zeros_like(pred));mse=error.square().sum((0,1))/n.clamp_min(1)
    negative=(torch.relu(-mu/sd-pred)*m).sum((0,1))/n.clamp_min(1)
    return ((mse+.1*negative)*has).sum()/has.sum().clamp_min(1)
