import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../ces_prediction')))

import torch
from model import MultimodalCESPredictor

def test_dry_run():
    print("[Dry-Run] Initializing model and dummy tensors...")
    try:
        model = MultimodalCESPredictor(window_size=10)
        
        # Batch=2, window=10
        bes = torch.randn(2, 10, 9)
        ecei = torch.randn(2, 10, 4)
        mc = torch.randn(2, 10, 2)
        dt = torch.randn(2, 10, 1) # Time-Encoding
        
        print("[Dry-Run] Running forward pass...")
        out = model(bes, ecei, mc, dt)
        
        print("[Dry-Run] Running backward pass...")
        loss = out.sum()
        loss.backward()
        
        print("[Dry-Run] Success! Model graph is physically and computationally valid.")
    except Exception as e:
        print(f"[Dry-Run] FAILED: {e}")
        exit(1) # 오류 발생 시 시스템에서 이를 캐치하도록 1 반환

if __name__ == "__main__":
    test_dry_run()