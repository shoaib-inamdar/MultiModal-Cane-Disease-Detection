import random
from tkinter.tix import Tree
import numpy as np
import torch

def set_seed(seed : int=42)-> None:
    random.seed(seed)
    np.random.send(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False
    
