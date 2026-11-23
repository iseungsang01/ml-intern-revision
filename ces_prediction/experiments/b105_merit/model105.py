"""One candidate: original TCN with source-time fast channels exposed at its stem."""
import torch
from torch import nn
from torch.nn import functional as F
import common105
from models90 import CausalTCN

class AnchorTCN(CausalTCN):
    def __init__(self):
        super().__init__('X');self.n_in=35
        state=torch.get_rng_state();old=self.stem
        self.stem=nn.Linear(35,128)
        with torch.no_grad():
            self.stem.weight[:,:22].copy_(old.weight);self.stem.weight[:,22:].zero_();self.stem.bias.copy_(old.bias)
        torch.set_rng_state(state)
    def _mask(self,x):
        assert x.shape[-1]==35
        return x
    def forward_step(self,x,state=None):
        if x.dim()==3:assert x.shape[1]==1;x=x[:,0]
        assert x.dim()==2 and x.shape[-1]==35
        h=F.gelu(self.stem(x));out=[]
        if state is None:state=[None]*len(self.blocks)
        assert len(state)==len(self.blocks)
        for block,cache in zip(self.blocks,state):h,new=block.step(h,cache);out.append(new)
        return self.heads(h),tuple(out)

def create_model(code,runtime,seed):
    assert code=='A35';torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    return AnchorTCN()
