
from torch import nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("device is",device);

import pickle

def saveRunDat(data):
	with open(outPath,"wb") as fp:
		pickle.dump(data,outPath)
	print("dumped to",outPath)