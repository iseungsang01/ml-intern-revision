"""Author-authorized checkpoint stop of this repository's old B85 allocation.

No training source is changed and no GPU benchmark is run. Exact process handles,
repository working directories, source/config hashes and a CPU-loaded atomic
checkpoint are verified before an owned training child is terminated nonzero.
"""
import argparse
import ctypes as C
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
CACHE=ROOT/'data/.b91'
B85=ROOT/'data/.b85'
SUITE=ROOT/'data/.queued_suite_20261122'
AUTHOR_CODE=0xE0000091
K=C.WinDLL('kernel32',use_last_error=True)
K.OpenProcess.argtypes=[C.c_ulong,C.c_int,C.c_ulong];K.OpenProcess.restype=C.c_void_p
K.CloseHandle.argtypes=[C.c_void_p]
K.GetExitCodeProcess.argtypes=[C.c_void_p,C.POINTER(C.c_ulong)]
K.TerminateProcess.argtypes=[C.c_void_p,C.c_uint]
K.WaitForSingleObject.argtypes=[C.c_void_p,C.c_ulong]
K.ReadProcessMemory.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_size_t,C.POINTER(C.c_size_t)]
N=C.WinDLL('ntdll')


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2),encoding='utf8');tmp.replace(path)


def sha(data):return hashlib.sha256(data).hexdigest()


def processes():
    command="Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object ProcessId,ParentProcessId,ExecutablePath,CommandLine | ConvertTo-Json -Compress"
    result=subprocess.run(['powershell','-NoProfile','-Command',command],capture_output=True,text=True,check=True)
    values=json.loads(result.stdout)
    return [values] if isinstance(values,dict) else values


def argv(command):
    shell=C.WinDLL('shell32');f=shell.CommandLineToArgvW
    f.argtypes=[C.c_wchar_p,C.POINTER(C.c_int)];f.restype=C.POINTER(C.c_wchar_p)
    n=C.c_int();pointer=f(command,C.byref(n));assert pointer,'Command line parse failed'
    try:return [pointer[i] for i in range(n.value)]
    finally:K.LocalFree(C.cast(pointer,C.c_void_p))


class Process:
    def __init__(self,pid,terminate=False):
        self.pid=int(pid)
        self.handle=K.OpenProcess(0x1000|0x0400|0x0010|0x100000|(0x0001 if terminate else 0),False,self.pid)
        if not self.handle:raise RuntimeError('Cannot verify process handle '+str(pid))
    def close(self):
        if self.handle:K.CloseHandle(self.handle);self.handle=None
    def code(self):
        code=C.c_ulong();assert K.GetExitCodeProcess(self.handle,C.byref(code))
        return int(code.value)
    def read(self,address,size):
        buf=C.create_string_buffer(size);got=C.c_size_t()
        assert K.ReadProcessMemory(self.handle,C.c_void_p(address),buf,size,C.byref(got)) and got.value==size,'Cannot read process identity'
        return buf.raw
    def cwd(self):
        assert C.sizeof(C.c_void_p)==8,'64-bit Windows identity reader required'
        class Basic(C.Structure):
            _fields_=[('reserved',C.c_void_p),('peb',C.c_void_p),('r2',C.c_void_p*2),('pid',C.c_size_t),('r3',C.c_void_p)]
        query=N.NtQueryInformationProcess;query.argtypes=[C.c_void_p,C.c_ulong,C.c_void_p,C.c_ulong,C.c_void_p]
        basic=Basic();assert query(self.handle,0,C.byref(basic),C.sizeof(basic),None)==0
        params=int.from_bytes(self.read(basic.peb+0x20,8),'little')
        value=self.read(params+0x38,16);length=int.from_bytes(value[:2],'little');address=int.from_bytes(value[8:16],'little')
        assert 0<length<32768 and address
        return Path(self.read(address,length).decode('utf-16-le')).resolve()
    def terminate(self):
        assert self.code()==259,'Process already exited; refusing a stale stop'
        assert K.TerminateProcess(self.handle,AUTHOR_CODE),'Owned process stop failed'
        assert K.WaitForSingleObject(self.handle,15000)==0,'Owned process did not exit'


def identify(row,relative,stop=False):
    assert Path(row['ExecutablePath']).resolve()==Path(sys.executable).resolve(),'Interpreter identity mismatch'
    words=argv(row['CommandLine']);scripts=[word for word in words if word.lower().endswith('.py')]
    assert len(scripts)==1,'Ambiguous Python script identity'
    script=Path(scripts[0]);script=(script if script.is_absolute() else ROOT/script).resolve()
    assert script==(ROOT/relative).resolve(),'Repository script identity mismatch'
    handle=Process(row['ProcessId'],stop)
    try:
        assert handle.code()==259 and handle.cwd()==ROOT,'Process working directory or liveness mismatch'
    except BaseException:handle.close();raise
    return handle,words


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args()
    CACHE.mkdir(parents=True,exist_ok=True)
    if (CACHE/'author_stop.json').exists():raise RuntimeError('Author stop record already exists; inspect before another control action')
    rows=processes();suite_state=json.loads((SUITE/'status.json').read_text())
    master=next(row for row in rows if row['ProcessId']==suite_state['pid'])
    assert suite_state['state']=='waiting_dependency'
    assert int((SUITE/'launch_pid.txt').read_text().strip())==master['ProcessId']
    assert int((SUITE/'suite.lock').read_text().strip())==master['ProcessId']
    mh,_=identify(master,'ces_prediction/experiments/queued_suite_20261122.py',stop=a.execute)
    if not a.execute:
        print(json.dumps({'master_pid':master['ProcessId'],'verified_cwd':str(mh.cwd()),'action':'No process changed; --execute required'}));mh.close();return
    # The source-independent controller never exposes this process to CUDA.
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    sys.path.insert(0,str(ROOT/'ces_prediction/experiments/b85_unified'))
    import torch
    import train85 as T
    assert not torch.cuda.is_available(),'Controller must remain CPU-only'
    pins=json.loads((ROOT/'ces_prediction/experiments/b88_anchor/frozen_b85_sources.json').read_text())
    for rel,expected in pins.items():assert sha((ROOT/rel).read_bytes())==expected,'Frozen source changed: '+rel
    completed=[]
    for folder in sorted((B85/'runs').iterdir()):
        if (folder/'complete.json').exists():assert T.verify_complete(folder);completed.append(folder.name)
    assert len(completed)>=4,'Prior clean completed fits missing'
    record={'reason':'Author budget revision, B85 addendum4; no scientific score readout',
            'time':time.strftime('%Y-%m-%dT%H:%M:%S'),'completed_clean_fits':completed,
            'master':master,'state':'stopping_idle_master'}
    save(CACHE/'author_stop_master.json',record)
    mh.terminate();mh.close() # Idle follow-up scheduler only, never a training process.
    deadline=time.monotonic()+120
    while time.monotonic()<deadline:
        rows=processes()
        children=[r for r in rows if 'train85.py' in (r['CommandLine'] or '')]
        assert len(children)<=1,'Multiple possible trainers; no process stopped'
        if not children:time.sleep(2);continue
        row=children[0];ch,words=identify(row,'ces_prediction/experiments/b85_unified/train85.py',True)
        try:
            assert '--smoke' not in words
            name=words[words.index('--unit')+1]+'__'+words[words.index('--init')+1]+'__'+words[words.index('--variant')+1]
            folder=B85/'runs'/name;config=json.loads((folder/'config.json').read_text())
            clean={k:v for k,v in config.items() if k!='config_hash'}
            assert T._config_hash(clean,config['unit'])==config['config_hash'],'Partial configuration/source mismatch'
            assert not (folder/'complete.json').exists(),'Fit completed naturally; rerun identity guard for successor'
            runner=next(r for r in rows if r['ProcessId']==row['ParentProcessId'])
            rh,_=identify(runner,'ces_prediction/experiments/b85_unified/run85.py')
            queue_state=json.loads((B85/'queue_status.json').read_text())
            queue=next(r for r in rows if r['ProcessId']==runner['ParentProcessId']==queue_state['pid'])
            qh,_=identify(queue,'ces_prediction/experiments/b85_unified/queue85.py')
            checkpoint=folder/'latest.pt'
            if not checkpoint.exists():rh.close();qh.close();time.sleep(2);continue
            raw=checkpoint.read_bytes()
            saved=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
            required=('epoch','model_state','optimizer_state','torch_rng','cuda_rng','np_rng','py_rng','history','config_hash')
            assert all(k in saved for k in required),'Checkpoint lacks resumable state'
            assert saved['config_hash']==config['config_hash'] and 0<=saved['epoch']<80
            assert saved['history'][-1]['epoch']==saved['epoch'] and saved['cuda_rng'] is not None
            def finite(obj):
                if torch.is_tensor(obj):return bool(torch.isfinite(obj).all())
                if isinstance(obj,dict):return all(finite(v) for v in obj.values())
                if isinstance(obj,(list,tuple)):return all(finite(v) for v in obj)
                if isinstance(obj,float):return math.isfinite(obj)
                return True
            assert finite(saved['model_state']) and finite(saved['optimizer_state']),'Nonfinite checkpoint state'
            assert finite(saved['history']),'Nonfinite loss history'
            preserved=CACHE/'author_stop_snapshot/latest_preserved.pt';preserved.parent.mkdir(parents=True,exist_ok=True)
            tmp=preserved.with_suffix('.tmp');tmp.write_bytes(raw);tmp.replace(preserved)
            record.update(state='checkpoint_verified_before_stop',trainer=row,run=name,epoch=int(saved['epoch']),
                          config_hash=config['config_hash'],checkpoint_sha256=sha(raw),
                          checkpoint_path=checkpoint.relative_to(ROOT).as_posix(),
                          preserved_checkpoint=preserved.relative_to(ROOT).as_posix(),
                          runner=runner,queue=queue)
            save(CACHE/'author_stop.json',record) # BEFORE terminating the verified owned trainer.
            if ch.code()!=259:raise RuntimeError('Trainer exited naturally during verification; no stop sent')
            ch.terminate();record['trainer_exit_code']=ch.code()
            for label,handle in (('runner',rh),('queue',qh)):
                result=K.WaitForSingleObject(handle.handle,30000)
                record[label+'_exited']=result==0
                record[label+'_exit_code']=handle.code();handle.close()
            assert record['runner_exited'] and record['queue_exited'],'Old supervisors have not unwound; do not start GPU probes'
            record['state']='superseded_by_author_revision';record['finished']=time.strftime('%Y-%m-%dT%H:%M:%S')
            save(CACHE/'author_stop.json',record)
            for path in (B85/'queue_status.json',SUITE/'status.json'):
                previous=json.loads(path.read_text())
                save(path,{'state':'superseded_by_author_revision','reason':record['reason'],
                           'record':'data/.b91/author_stop.json','updated':record['finished'],'previous':previous})
            print(json.dumps({'state':record['state'],'stopped_trainer_pid':ch.pid,'epoch':record['epoch'],
                              'completed_clean_fits':len(completed),'runner_and_queue_exited':True}))
            return
        finally:ch.close()
    raise RuntimeError('No verified atomic checkpoint within120s; training child was not stopped')


if __name__=='__main__':main()
