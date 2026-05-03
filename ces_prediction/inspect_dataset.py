import sys
import os
from pathlib import Path

# Add the ces_prediction directory to sys.path to import dataset
sys.path.append(str(Path(r"c:\Users\lss\Documents\GitHub\ml-intern-revision\ces_prediction")))

from dataset import KSTAR_CES_Dataset

def inspect():
    data_dir = Path(r"c:\Users\lss\Documents\GitHub\ml-intern-revision\data")
    window_size = 10
    dataset = KSTAR_CES_Dataset(data_dir=data_dir, window_size=window_size)
    
    print(f"Total samples loaded: {len(dataset)}")
    print(f"Feature dimensions: {dataset.feature_dims}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print("\n--- Sample 0 ---")
        print(f"File: {sample['file']}")
        print(f"Row Index: {sample['row_index']}")
        print(f"Target (CES_TI, CES_VT): {sample['target'].tolist()}")
        print(f"BES shape: {sample['bes'].shape}")
        print(f"ECEI shape: {sample['ecei'].shape}")
        print(f"MC shape: {sample['mc'].shape}")
        print(f"Time features (first 3 steps):\n{sample['time_features'][:3]}")
        
        # Print a slice of BES data to see actual values
        print(f"\nBES values (first 3 steps, first 5 channels):\n{sample['bes'][:3, :5]}")
    else:
        print("No samples found in the dataset.")

if __name__ == "__main__":
    inspect()
