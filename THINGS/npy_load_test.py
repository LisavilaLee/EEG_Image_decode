import numpy as np

file_path = r"/userhome2/liweile/EEG_Image_decode/THINGS/Preprocessed_data_250Hz/sub-01/preprocessed_eeg_test.npy"
try:
    data = np.load(file_path, allow_pickle=True)
    print("File loaded successfully")
except Exception as e:
    print(f"Error loading file: {e}")