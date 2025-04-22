import numpy as np

file_path = r"/userhome2/liweile/EEG_Image_decode/datasets--LidongYang--EEG_Image_decode/blobs/1ddd73ef392883004c384cda1fdf7ef7a95e2078fba8e49293bcf381c1c762c2.incomplete"
try:
    data = np.load(file_path, allow_pickle=True)
    print("File loaded successfully")
except Exception as e:
    print(f"Error loading file: {e}")