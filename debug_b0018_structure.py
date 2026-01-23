import scipy.io
import pandas as pd
import numpy as np

def inspect_structure(file_path):
    mat = scipy.io.loadmat(file_path)
    # Usually mat['B0018']
    key = [k for k in mat.keys() if not k.startswith('__')][0]
    print(f"Dataset Key: {key}")
    
    data = mat[key][0, 0]['cycle'][0]
    print(f"Total entries in 'cycle': {len(data)}")
    
    for i in range(min(5, len(data))):
        print(f"\n--- Entry {i} ---")
        cycle = data[i]
        print(f"Fields: {cycle.dtype.names}")
        if 'type' in cycle.dtype.names:
            print(f"Type: {cycle['type'][0]}")
        if 'time' in cycle.dtype.names:
            print(f"Time (raw): {cycle['time']}")
            
file_path = 'NASA_PCOE_DATA/B0018.mat'
inspect_structure(file_path)
