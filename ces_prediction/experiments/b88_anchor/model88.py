"""Claude-origin shared encoder; corrected baseline constructor compatibility."""
import common88
import torch
from torch import nn
from model85 import B85Model
class B88Model(B85Model):
    def __init__(self,variant='A'):
        assert variant in ('A','C')
        super().__init__(variant=variant,n_in=39,n_fast=32)
        self.phi=nn.Sequential(nn.Linear(13,32),nn.GELU(),nn.Linear(32,16),nn.GELU())
    def _fuse_input(self,x):
        assert x.shape[-1]==35
        current=self.phi(x[...,:13])
        anchor=self.phi(x[...,22:35])*x[...,18:19] if self.variant=='A' else torch.zeros_like(current)
        return torch.cat((current,anchor,x[...,15:22]),dim=-1)
    def forward(self,x,lengths=None):return super().forward(self._fuse_input(x),lengths)
    def forward_step(self,x,state=None):return super().forward_step(self._fuse_input(x),state)
def create_model(variant='A',init_seed=1701):
    torch.manual_seed(init_seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(init_seed)
    m=B88Model(variant);assert m.param_count()<1000000;return m
