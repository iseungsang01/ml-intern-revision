import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from dataset import KSTAR_CES_Dataset
from model import MultimodalCESPredictor

# [Phase 1] Import wandb for experiment tracking
try:
    import wandb
except ImportError:
    pass # wandb가 없는 환경일 경우 무시

def train():
    # 1. Hyperparameters
    DATA_DIR = "../data"
    WINDOW_SIZE = 10
    BATCH_SIZE = 64
    EPOCHS = 10
    LR = 0.001
    
    # Init W&B (Offline mode 기본 설정으로 안전하게 실행)
    if 'wandb' in globals():
        wandb.init(project="kstar-ces-prediction", mode="disabled", config={
            "window_size": WINDOW_SIZE, "batch_size": BATCH_SIZE, "epochs": EPOCHS, "lr": LR
        })
    
    # 2. Prepare Data
    print("Initializing dataset...")
    full_dataset = KSTAR_CES_Dataset(data_dir=DATA_DIR, window_size=WINDOW_SIZE)
    
    if len(full_dataset) == 0:
        print("Error: No valid data found.")
        return
        
    # 계통 추출(Systematic Sampling): 매 5번째 샘플을 검증(Validation) 셋으로 할당 (20%)
    total_samples = len(full_dataset)
    val_indices = list(range(0, total_samples, 5))
    train_indices = [i for i in range(total_samples) if i % 5 != 0]
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Model, Loss, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = MultimodalCESPredictor(window_size=WINDOW_SIZE).to(device)
    criterion = nn.MSELoss()  
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # 4. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            bes = batch['bes'].to(device)
            ecei = batch['ecei'].to(device)
            mc = batch['mc'].to(device)
            dt = batch['dt'].to(device)
            targets = batch['target'].to(device)
            
            optimizer.zero_grad()
            outputs = model(bes, ecei, mc, dt)
            
            loss_mse = criterion(outputs, targets)
            
            # [Phase 3] PINN (Physics-Informed) Loss
            # 물리적 제약: 플라즈마 이온 온도(T_i)는 음수일 수 없습니다. (outputs[:, 0]은 T_i 예측값)
            # 만약 모델이 음수를 예측하면 강한 페널티를 부여합니다.
            penalty_neg_ti = torch.relu(-outputs[:, 0]).mean()
            loss = loss_mse + 0.1 * penalty_neg_ti
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * bes.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                bes = batch['bes'].to(device)
                ecei = batch['ecei'].to(device)
                mc = batch['mc'].to(device)
                dt = batch['dt'].to(device)
                targets = batch['target'].to(device)
                
                outputs = model(bes, ecei, mc, dt)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * bes.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if 'wandb' in globals():
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})
        
    # 5. Save model and metrics
    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), "weights/multimodal_ces.pth")
    print("Training complete! Model saved to weights/multimodal_ces.pth")
    
    # Save metrics for Evaluation Agent
    metrics = {
        "final_train_loss": train_loss,
        "final_val_loss": val_loss,
        "epochs": EPOCHS
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f)
    print("Metrics saved to metrics.json")
    
    if 'wandb' in globals():
        wandb.finish()

if __name__ == "__main__":
    train()
