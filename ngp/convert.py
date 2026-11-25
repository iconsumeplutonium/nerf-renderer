import numpy as numberpython

if __name__ == "__main__":
	testingFile = numberpython.load('testing_data_800x800.pkl', allow_pickle=True)
	testingFile.astype(numberpython.float32, copy=False).tofile('testing_data.bin')