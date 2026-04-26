import torch
import torch.nn as nn

class MultimodalCESPredictor(nn.Module):
    def __init__(self, window_size=10):
        super(MultimodalCESPredictor, self).__init__()
        
        # 1. BES Feature Extractor (1D CNN over time for 9 spatial channels)
        # Input shape: (Batch, window_size, 9) -> (Batch, 9, window_size) for Conv1d
        self.bes_extractor = nn.Sequential(
            nn.Conv1d(in_channels=9, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * window_size, 64)
        )
        
        # 2. ECEI Feature Extractor (1D CNN over time for 4 spatial channels)
        # Note: If ECEI is mapped to a 2D grid later, this can easily be replaced with Conv3d or Conv2d
        self.ecei_extractor = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * window_size, 64)
        )
        
        # 3. Mirnov Coil (MC) Extractor (1D CNN instead of LSTM)
        # Input shape: (Batch, window_size, 2) -> (Batch, 2, window_size) for Conv1d
        self.mc_extractor = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * window_size, 64)
        )
        
        # 4. [Phase 3] Time-Encoding Extractor
        # Input shape: (Batch, window_size, 1) -> (Batch, 1, window_size)
        self.time_extractor = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * window_size, 16)
        )
        
        # 5. Late Fusion Module
        # Concatenate features: 64(BES) + 64(ECEI) + 64(MC) + 16(dt) = 208
        self.fusion_fc = nn.Sequential(
            nn.Linear(208, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Output: [CES_TI, CES_VT]
        )
        
    def forward(self, bes, ecei, mc, dt):
        # bes/ecei/mc/dt shape: (B, window_size, channels) 
        # -> permute to (B, channels, window_size) for Conv1d
        bes = bes.permute(0, 2, 1)
        ecei = ecei.permute(0, 2, 1)
        mc = mc.permute(0, 2, 1)
        dt = dt.permute(0, 2, 1)
        
        # Extract features
        bes_feat = self.bes_extractor(bes)      # (B, 64)
        ecei_feat = self.ecei_extractor(ecei)   # (B, 64)
        mc_feat = self.mc_extractor(mc)         # (B, 64)
        dt_feat = self.time_extractor(dt)       # (B, 16)
        
        # Late Fusion
        fused = torch.cat((bes_feat, ecei_feat, mc_feat, dt_feat), dim=1) # (B, 208)
        output = self.fusion_fc(fused)                           # (B, 2)
        
        return output

if __name__ == "__main__":
    # Test Forward Pass
    model = MultimodalCESPredictor(window_size=10)
    
    dummy_bes = torch.randn(8, 10, 9)   # Batch=8, window=10, channels=9
    dummy_ecei = torch.randn(8, 10, 4)  # Batch=8, window=10, channels=4
    dummy_mc = torch.randn(8, 10, 2)    # Batch=8, window=10, channels=2
    
    preds = model(dummy_bes, dummy_ecei, dummy_mc)
    print(f"Output shape: {preds.shape}") # Expected: (8, 2)
