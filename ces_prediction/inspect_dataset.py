import sys
from pathlib import Path
import torch
import pandas as pd
import numpy as np

# Add the ces_prediction directory to sys.path to import dataset
sys.path.append(str(Path(__file__).resolve().parent))

from dataset import KSTAR_CES_Dataset

class LimitedDataset(KSTAR_CES_Dataset):
    """Dataset that only looks at the first few files for speed."""
    def __init__(self, *args, max_files=1, **kwargs):
        self.max_files = max_files
        super().__init__(*args, **kwargs)
    
    def _build_index(self):
        original_files = self.files
        self.files = self.files[:self.max_files]
        indices = super()._build_index()
        self.files = original_files
        return indices

def inspect_single_sample_details():
    # ==========================================
    # 아래 설정값들을 바꿔보며 테스트하세요!
    # ==========================================
    WINDOW_SIZE = 10                  # 전체 윈도우(텐서) 크기
    MIN_SUBSET_SIZE = 3               # 최소 데이터 개수
    AUGMENTATION = True               # 증강(부분집합 생성) 사용 여부
    SAMPLE_INDEX = 0                  # 보고 싶은 샘플의 인덱스 번호
    # ==========================================

    data_dir = Path(__file__).resolve().parents[1] / "data"
    
    dataset = LimitedDataset(
        data_dir=data_dir, 
        window_size=WINDOW_SIZE, 
        temporal_subset_augmentation=AUGMENTATION, 
        min_subset_size=MIN_SUBSET_SIZE
    )
    
    if len(dataset) <= SAMPLE_INDEX:
        print(f"Error: Dataset length is {len(dataset)}, but you requested index {SAMPLE_INDEX}")
        return

    sample = dataset[SAMPLE_INDEX]

    print(f"\n{'='*30}")
    print(f" INSPECTION: Index {SAMPLE_INDEX}")
    print(f"{'='*30}")
    print(f"File: {Path(sample['file']).name}")
    print(f"Target Row: {sample['row_index']}")
    print(f"Included Rows: {sample['row_indices'][sample['input_mask']].tolist()}")
    print(f"Input Mask (1=Data, 0=Padding): {sample['input_mask'].int().tolist()}")
    print(f"Target Value: {sample['target'].tolist()}")
    
    # 텐서 정보 요약 출력
    def print_tensor_summary(name, tensor):
        valid_len = sample['input_mask'].sum().item()
        print(f"\n[{name}] Shape: {tuple(tensor.shape)}")
        print(f"  - Valid rows (up to index {valid_len-1}):\n{tensor[:valid_len]}")
        if valid_len < WINDOW_SIZE:
            print(f"  - Padded rows (index {valid_len} to {WINDOW_SIZE-1}): [ALL ZEROS]")

    print_tensor_summary("BES Sensor", sample['bes'])
    
    print(f"\n[CES History] Shape: {tuple(sample['ces_history'].shape)}")
    print(f"  (TI, VT, Observed_Mask)")
    print(sample['ces_history'])

if __name__ == "__main__":
    inspect_single_sample_details()
