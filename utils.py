import json, pickle
import time
import torch
import numpy as np 


def torch2npy(tensor):
    if len(tensor.shape) == 4:
        tensor = tensor.unsqueeze(0)
    npy = tensor.detach().cpu().numpy()
    return npy

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_json(fname, encoding='utf-8'):
    with open(fname, encoding=encoding) as f:
        fn = json.load(f)
    return fn

def save_pkl(file, fname):
    with open(f'./{fname}.pkl', mode='wb') as f:
        pickle.dump(file, f)
    print(f'Success Saving File! PATH: ./{fname}.pkl')

def load_pkl(path):
    with open(path, mode='rb') as f:
        file = pickle.load(f)
    return file 

def elapsed_time(start, end):
    elapsed = end - start 
    elapsed_min = elapsed // 60 
    elapsed_sec = elapsed - elapsed_min * 60 
    return elapsed_min, elapsed_sec 
