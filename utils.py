import json 

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
