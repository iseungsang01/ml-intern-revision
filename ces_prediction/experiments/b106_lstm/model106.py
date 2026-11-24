"""Original SeqCESLSTMv2 temperature branch; no replacement by a GP model."""
import torch
from torch import nn
import common106
import sys
sys.path.insert(0,str(common106.HERE.parent/'seq'))
from model_seq_v2 import SeqCESLSTMv2
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
class TemperatureLSTM(nn.Module):
    def __init__(self,dense=True):
        super().__init__();self.dense=dense
        b=SeqCESLSTMv2(n_in=22,n_fast=15,n_extra_vt=0,n_extra_ti=0)
        self.lstm_ti=b.lstm_ti;self.norm_ti=b.norm_ti;self.head_ti=b.head_ti
    def forward(self,x,lengths=None):
        x=x[...,:22]
        if lengths is not None and not self.dense:
            p=pack_padded_sequence(x,lengths.cpu(),batch_first=True,enforce_sorted=False);h,_=self.lstm_ti(p);h,_=pad_packed_sequence(h,batch_first=True,total_length=x.shape[1])
        else:h,_=self.lstm_ti(x)
        f=self.head_ti(self.norm_ti(h));return torch.cat([f,f*0],dim=-1)
    def forward_step(self,x,state=None):
        if x.dim()==2:x=x.unsqueeze(1)
        h,state=self.lstm_ti(x[...,:22],state);f=self.head_ti(self.norm_ti(h)).squeeze(1);return torch.cat([f,f*0],dim=-1),state
def create_model(code,runtime,seed):
    assert code=='LSTM_F';torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    return TemperatureLSTM(runtime['dense'])
