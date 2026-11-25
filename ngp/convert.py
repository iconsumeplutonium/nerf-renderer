import numpy as numberpython
import torch
from torch import Tensor
from collections import OrderedDict
import os

if __name__ == "__main__":
	testingFile = numberpython.load('testing_data_800x800.pkl', allow_pickle=True)
	testingFile.astype(numberpython.float32, copy=False).tofile('testing_data.bin')

	os.makedirs('weights', exist_ok=True)
	stateDict: OrderedDict[str, Tensor] = torch.load('../igp/ngp_weights.pt', map_location='cpu')
	for name, t in stateDict.items():
		t.detach().contiguous().numpy().astype('<f4').tofile(f'weights/{name}.bin')
		print(name, tuple(t.shape))