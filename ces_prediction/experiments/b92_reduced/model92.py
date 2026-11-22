"""Only the three registered configurations and selected paired T control."""
import torch
from torch import nn
import common92 as M
from model88 import create_model as anchor_model
from models90 import create_model as candidate_model
from runtime92 import configure
class Model(nn.Module):
    def __init__(self,code,runtime,seed):
        super().__init__();assert code in ('A','C','T','TX','TH');self.code=code;self.runtime=runtime
        self.base=anchor_model(code,seed) if code in ('A','C') else candidate_model('T','H' if code=='TH' else 'X',seed)
        configure(runtime)
    def forward(self,x,lengths=None):
        if self.code not in ('A','C'):return self.base(x[...,:22],lengths)
        if not self.runtime['dense']:return self.base(x,lengths)
        h,_=self.base.lstm_ti(self.base._fuse_input(x));h=self.base.norm_ti(h)
        return torch.cat((self.base.head_F(h),self.base.head_L5(h)),dim=-1)
    def forward_step(self,x,state=None):return self.base.forward_step(x if self.code in ('A','C') else x[...,:22],state)
    def param_count(self):return sum(p.numel() for p in self.parameters())
def create_model(code,runtime,seed):return Model(code,runtime,seed)
