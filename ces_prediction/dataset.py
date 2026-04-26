import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from functools import lru_cache

class KSTAR_CES_Dataset(Dataset):
    def __init__(self, data_dir, window_size=10, is_train=True):
        """
        KSTAR Multimodal Dataset (Lazy-Loading & Time-Encoding version)
        - 메모리 효율성을 위해 파일의 메타(인덱스)만 미리 읽고, __getitem__에서 필요 시점에 Load합니다.
        """
        self.window_size = window_size
        self.data_dir = data_dir
        self.files = glob.glob(os.path.join(data_dir, "*.csv"))
        
        self.sample_indices = [] # (file_path, valid_target_index)
        
        print(f"Indexing {len(self.files)} files for Lazy Loading...")
        for file in self.files:
            try:
                # 인덱싱 과정에서는 메모리 최소화를 위해 타겟 컬럼만 가볍게 스캔합니다.
                df_targets = pd.read_csv(file, usecols=['CES_TI', 'CES_VT'])
                valid_mask = df_targets.notnull().all(axis=1)
                valid_indices = df_targets[valid_mask].index
                
                for idx in valid_indices:
                    if idx >= self.window_size - 1:
                        self.sample_indices.append((file, idx))
            except Exception as e:
                pass

    @lru_cache(maxsize=20)
    def _get_file_data(self, file_path):
        # LRU Cache를 사용해 최근 접근한 20개 파일의 DataFrame만 메모리에 상주시킵니다.
        return pd.read_csv(file_path)

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, i):
        file_path, idx = self.sample_indices[i]
        df = self._get_file_data(file_path)
        
        bes_cols = [c for c in df.columns if c.startswith('BES_')]
        ecei_cols = [c for c in df.columns if c.startswith('ECEI_')]
        mc_cols = [c for c in df.columns if c.startswith('MC1')]
        
        # 윈도우 슬라이싱
        window_df = df.iloc[idx - self.window_size + 1 : idx + 1]
        
        # 만약 윈도우 안에 NaN이 있다면 방어 로직 (0으로 패딩하거나 안전한 값 처리)
        # 이미 인덱싱에서 걸렀지만 센서 피처 자체에 NaN이 낄 수 있으므로 0 채우기
        window_df = window_df.fillna(0.0)
        
        bes = torch.tensor(window_df[bes_cols].values, dtype=torch.float32)
        ecei = torch.tensor(window_df[ecei_cols].values, dtype=torch.float32)
        mc = torch.tensor(window_df[mc_cols].values, dtype=torch.float32)
        
        # [Phase 3] Time-Encoding: dt 추출
        # 측정 시간 불규칙성 패턴을 모델이 인식하도록, 윈도우 시작점 대비 현재 시간의 상대적 변화량(dt)을 입력
        time_vals = window_df['time'].values
        dt = time_vals - time_vals[0]
        dt = torch.tensor(dt, dtype=torch.float32).unsqueeze(1) # (window_size, 1) 차원으로 확장
        
        target_val = df.iloc[idx][['CES_TI', 'CES_VT']].values
        target = torch.tensor(target_val, dtype=torch.float32)
        
        return {'bes': bes, 'ecei': ecei, 'mc': mc, 'dt': dt, 'target': target}

if __name__ == "__main__":
    # Test DataLoader
    dataset = KSTAR_CES_Dataset(data_dir="../data", window_size=10)
    print(f"Total samples generated: {len(dataset)}")
    if len(dataset) > 0:
        sample = dataset[0]
        print("BES shape:", sample['bes'].shape)     # (window_size, 9)
        print("ECEI shape:", sample['ecei'].shape)   # (window_size, 4)
        print("MC shape:", sample['mc'].shape)       # (window_size, 2)
        print("Target shape:", sample['target'].shape) # (2,)
