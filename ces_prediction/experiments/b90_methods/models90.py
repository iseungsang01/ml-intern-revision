"""Registered candidate building blocks, not reproductions of named prior models.

Every candidate consumes the frozen 22-channel input and directly estimates the
same normalized F/L5 outputs. X/H are the original fast-enabled/first15-zeroed
variants. Losses, data treatment, splits and training budget belong to the adapter.
"""
import math,sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'b85_unified'))
from model85 import B85Model

METHODS=('G','T','R','C','U','V','N')
ARCHITECTURES={
 'G':{'name':'lag-conditioned decay GRU','hidden':160,'layers':2,'decay_condition_slots':[15,17,18],'dropout':.1},
 'T':{'name':'causal dilated TCN','width':128,'kernel':3,'dilations':[1,2,4,8,16,32],'receptive_field_rows':127,'dropout':.1},
 'R':{'name':'diagonal discrete learned observer','latent':128,'initial_transition':.9,'initial_process_softplus_argument':-4.,'variance_floor':1e-4},
 'C':{'name':'diagonal continuous-time learned observer','latent':128,'initial_rate_per_second':10.,'initial_process_softplus_argument':-4.,'variance_floor':1e-4},
 'U':{'name':'inherited B85 LSTM with external Huber loss','delta_normalized':1.},
 'V':{'name':'inherited B85 LSTM with external Student-t loss','df':4.,'scale_normalized':1.},
 'N':{'name':'inherited B85 LSTM with Gaussian mean/logvariance heads','logvariance_bounds':[-8.,8.]},
}

def _step_input(x):
    if x.dim()==3:
        assert x.shape[1]==1;x=x[:,0]
    assert x.dim()==2 and x.shape[-1]==22
    return x

def _dt(x):
    # Slot15 is log1p(previous-grid dt in seconds), zero on segment reset.
    return torch.expm1(x[:,15:16].clamp(0,math.log(61.)))

class DirectHeads(nn.Module):
    def __init__(self,width):
        super().__init__();self.norm=nn.LayerNorm(width)
        self.head_F=nn.Sequential(nn.Linear(width,64),nn.GELU(),nn.Linear(64,1))
        self.head_L5=nn.Sequential(nn.Linear(width,64),nn.GELU(),nn.Linear(64,1))
    def forward(self,h):
        h=self.norm(h);return torch.cat((self.head_F(h),self.head_L5(h)),dim=-1)

class Candidate(nn.Module):
    def __init__(self,method,variant):
        super().__init__();assert variant in ('X','H');self.method=method;self.variant=variant;self.n_in=22;self.n_fast=15
    def _mask(self,x):
        assert x.shape[-1]==22
        if self.variant=='H':x=x.clone();x[...,:15]=0
        return x
    def param_count(self):return sum(p.numel() for p in self.parameters())

class RecurrentCandidate(Candidate):
    def forward(self,x,lengths=None):
        assert x.dim()==3;x=self._mask(x);state=None;out=[]
        lens=None if lengths is None else lengths.to(x.device)
        for j in range(x.shape[1]):
            h,state=self._step(x[:,j],state)
            if lens is not None:h=h*(j<lens).unsqueeze(-1)
            out.append(self.heads(h))
        assert out,'Empty input segment'
        return torch.stack(out,dim=1)
    def forward_step(self,x,state=None):
        h,state=self._step(self._mask(_step_input(x)),state)
        return self.heads(h),state

class DecayGRU(RecurrentCandidate):
    """GRU-D-inspired hidden decay; no per-channel missingness imputation."""
    def __init__(self,variant):
        super().__init__('G',variant)
        self.cells=nn.ModuleList([nn.GRUCell(22,160),nn.GRUCell(160,160)])
        self.rates=nn.ModuleList([nn.Linear(3,160),nn.Linear(3,160)])
        self.dropout=nn.Dropout(.1);self.heads=DirectHeads(160)
    def _step(self,x,state):
        if state is None:state=(x.new_zeros((len(x),160)),x.new_zeros((len(x),160)))
        condition=x[:,[15,17,18]];dt=_dt(x);next_state=[];current=x
        for i,(cell,rate,previous) in enumerate(zip(self.cells,self.rates,state)):
            decay=torch.exp(-(F.softplus(rate(condition))*dt).clamp(max=80))
            h=cell(current,decay*previous);next_state.append(h)
            current=self.dropout(h) if i==0 else h
        return h,tuple(next_state)

class CausalBlock(nn.Module):
    def __init__(self,dilation):
        super().__init__();self.dilation=dilation;self.history=2*dilation
        self.norm=nn.LayerNorm(128);self.conv=nn.Conv1d(128,128,3,dilation=dilation);self.dropout=nn.Dropout(.1)
    def forward(self,x):
        z=F.gelu(self.norm(x)).transpose(1,2)
        z=self.conv(F.pad(z,(self.history,0))).transpose(1,2)
        return x+self.dropout(F.gelu(z))
    def step(self,x,cache):
        z=F.gelu(self.norm(x)).unsqueeze(-1)
        if cache is None:cache=z.new_zeros((len(z),128,self.history))
        window=torch.cat((cache,z),dim=-1)
        z=F.conv1d(window,self.conv.weight,self.conv.bias,dilation=self.dilation).squeeze(-1)
        return x+self.dropout(F.gelu(z)),window[:,:,-self.history:]

class CausalTCN(Candidate):
    """Six one-convolution residual blocks; 127-row bounded past context."""
    receptive_field_rows=127
    def __init__(self,variant):
        super().__init__('T',variant);self.stem=nn.Linear(22,128)
        self.blocks=nn.ModuleList(CausalBlock(d) for d in (1,2,4,8,16,32));self.heads=DirectHeads(128)
    def forward(self,x,lengths=None):
        assert x.dim()==3;h=F.gelu(self.stem(self._mask(x)))
        for block in self.blocks:h=block(h)
        if lengths is not None:
            valid=torch.arange(x.shape[1],device=x.device)[None,:]<lengths.to(x.device)[:,None];h=h*valid[...,None]
        return self.heads(h)
    def forward_step(self,x,state=None):
        h=F.gelu(self.stem(self._mask(_step_input(x))));caches=[]
        if state is None:state=[None]*len(self.blocks)
        assert len(state)==len(self.blocks)
        for block,cache in zip(self.blocks,state):h,new=block.step(h,cache);caches.append(new)
        return self.heads(h),tuple(caches)

class DiagonalObserver(RecurrentCandidate):
    """Learned latent innovation update, not a calibrated physical Kalman filter.

    R uses one discrete transition per row. C uses an exact stable diagonal
    exponential transition and elapsed-time process variance. Both learn latent
    pseudo-observations, not an observation operator for a physical state.
    """
    def __init__(self,method,variant):
        assert method in ('R','C');super().__init__(method,variant)
        self.drive=nn.Linear(22,128);self.observation=nn.Linear(22,128)
        self.obs_noise=nn.Linear(7,128)
        self.transition=nn.Parameter(torch.full((128,),math.atanh(.9) if method=='R' else math.log(math.expm1(10.))))
        self.process=nn.Parameter(torch.full((128,),-4.));self.heads=DirectHeads(128)
    def _step(self,x,state):
        if state is None:state=(x.new_zeros((len(x),128)),x.new_ones((len(x),128)))
        mean,var=state;drive=torch.tanh(self.drive(x));obs=torch.tanh(self.observation(x))
        q=F.softplus(self.process)+1e-4
        if self.method=='R':
            a=torch.tanh(self.transition);prior=a*mean+(1-a.abs())*drive;pv=a.square()*var+q
        else:
            a=torch.exp(-((F.softplus(self.transition)+1e-4)*_dt(x)).clamp(max=80))
            prior=a*mean+(1-a)*drive;pv=a.square()*var+q*(1-a.square())
        noise=F.softplus(self.obs_noise(x[:,15:22]))+1e-4
        gain=pv/(pv+noise);mean=prior+gain*(obs-prior)
        var=((1-gain)*pv).clamp(min=1e-8,max=1e4)
        return mean,(mean,var)

class GaussianLSTM(B85Model):
    """Mean return contract is unchanged; loss reads graph-bearing latest_logvar."""
    def __init__(self,variant='X'):
        super().__init__(variant=variant);self.method='N';self.latest_logvar=None
        self.logvar_F=nn.Sequential(nn.Linear(160,64),nn.GELU(),nn.Linear(64,1))
        self.logvar_L5=nn.Sequential(nn.Linear(160,64),nn.GELU(),nn.Linear(64,1))
    def _outputs(self,h):
        mean=torch.cat((self.head_F(h),self.head_L5(h)),dim=-1)
        self.latest_logvar=torch.cat((self.logvar_F(h),self.logvar_L5(h)),dim=-1).clamp(-8,8)
        return mean
    def forward(self,x,lengths=None):
        x=self._mask_input(x);assert x.dim()==3 and x.shape[-1]==22
        if lengths is None:h,_=self.lstm_ti(x)
        else:
            packed=pack_padded_sequence(x,lengths.cpu().to(torch.int64),batch_first=True,enforce_sorted=False)
            h,_=self.lstm_ti(packed);h,_=pad_packed_sequence(h,batch_first=True,total_length=x.shape[1])
        return self._outputs(self.norm_ti(h))
    def forward_step(self,x,state=None):
        x=self._mask_input(_step_input(x)).unsqueeze(1);h,state=self.lstm_ti(x,state)
        mean=self._outputs(self.norm_ti(h)).squeeze(1)
        self.latest_logvar=self.latest_logvar.squeeze(1)
        return mean,state

def create_model(method='G',variant='X',init_seed=1701):
    assert method in METHODS or method=='L'
    torch.manual_seed(init_seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(init_seed)
    if method=='G':model=DecayGRU(variant)
    elif method=='T':model=CausalTCN(variant)
    elif method in ('R','C'):model=DiagonalObserver(method,variant)
    elif method=='N':model=GaussianLSTM(variant)
    else:model=B85Model(variant=variant);model.method=method
    assert model.param_count()<1000000
    return model

build_model=create_model
def infer_batch(model,x,lengths=None):
    model.eval()
    with torch.no_grad():return model(x,lengths)
