# Grabs images from lfw-deepfunneled, resizes them to 105x105 (the centre part), binarizes them, and saves them as numpy arrays into binary files.
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def preprocess_lfw(input_dir,use_binarization=False):
    if not os.path.exists(input_dir):
        print(f"input directory {input_dir} does not exist.")
        return
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.jpg'):
                img_path = os.path.join(root, file)
                if use_binarization:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    img_cropped = img[60:190, 60:190] # This crop size was chosen to focus on the face area based on trial-error
                    img_resized = cv2.resize(img_cropped, (105, 105))
                    _, img_resized = cv2.threshold(img_resized, 128, 255, cv2.THRESH_BINARY)
                else:
                    img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
                    img_cropped = img[60:190, 60:190] # This crop size was chosen to focus on the face area based on trial-erro
                    img_resized = cv2.resize(img_cropped, (105, 105))
                np_img = np.array(img_resized, dtype=np.uint8)
                np.save(root + '/' + file.split('.')[0] + '.npy', np_img)



if __name__ == "__main__":
    input_directory = os.path.join('TrainingSet')
    preprocess_lfw(input_directory)