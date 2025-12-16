import tensorflow as tf
from tensorflow.keras import layers, models, Model
from tensorflow.keras.saving import register_keras_serializable
import numpy as np
import cv2
import os
import pandas as pd
import argparse
CSV = os.path.join('TrainingSet', 'lfw_pairs.csv')
BATCH_SIZE = 32
IMAGE_SHAPE = (105, 105, 3) 
MODEL = 'siamese_model.keras'
IMG1 = os.path.join('ValidationSet', 'RawFaces', 'Face Dataset', '1676', '2.jpg')
IMG2 = os.path.join('ValidationSet', 'RawFaces', 'Face Dataset', '1676', '3.jpg')


IMG3 = os.path.join('ValidationSet', 'RawFaces', 'Face Dataset', '1678', '1.jpg')
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)



@register_keras_serializable()
def l2_norm(x):
    return tf.math.l2_normalize(x, axis=1)

@register_keras_serializable()
def euclidean_distance(vects):
    x, y = vects
    sum_square = tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)
    return tf.sqrt(tf.maximum(sum_square, 1e-7))
def preprocess_real_image(image_path):
    img = cv2.imread(image_path)
    faces = face_classifier.detectMultiScale(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), scaleFactor=1.2, minNeighbors=10, minSize=(30, 30)
        )
    if len(faces) == 0:
        raise ValueError("Error: No face detected in the image.")
    if len(faces) > 1:
        raise ValueError(f"Error: Multiple faces detected ({len(faces)}). Please provide an image with a single person.")
    x, y, w, h = faces[0]

    padding = 10
    x_start = max(0, x - padding)
    y_start = max(0, y - padding)
    x_end = min(img.shape[1], x + w + padding)
    y_end = min(img.shape[0], y + h + padding)
    img_cropped = img[y_start:y_end, x_start:x_end]
    img_resized = cv2.resize(img_cropped, (105, 105))
    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_normalized = np.expand_dims(img_normalized, axis=0)
    return img_normalized


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img_cropped = img[60:190, 60:190]  # Crop to focus on face area
    img_resized = cv2.resize(img_cropped, (105, 105))
    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_normalized = np.expand_dims(img_normalized, axis=0)
    return img_normalized


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Face Comparison")
    group = parser.add_mutually_exclusive_group()
    
    group.add_argument('-custom', nargs=2, metavar=('PATH1', 'PATH2'), help='Compare two specific image files')
    group.add_argument('-same', action='store_true', help='Compare IMG1 and IMG2 (Hardcoded Same Person)')
    group.add_argument('-different', action='store_true', help='Compare IMG1 and IMG3 (Hardcoded Different People)')
    args = parser.parse_args()

    if args.custom:
        print(f"--- Running in Custom Mode ---")
        path_to_img1 = args.custom[0]
        path_to_img2 = args.custom[1]
    elif args.same:
        print(f"--- Testing Same Images ---")
        path_to_img1 = IMG1
        path_to_img2 = IMG2
    elif args.different:
        print(f"--- Testing Different Images ---")
        path_to_img1 = IMG1
        path_to_img2 = IMG3
    
    model = tf.keras.models.load_model(MODEL,custom_objects={'euclidean_distance': euclidean_distance}, compile=False, safe_mode=False)
    img1 = preprocess_real_image(path_to_img1)
    img2 = preprocess_real_image(path_to_img2)
    distance = model.predict([img1, img2])
    distance_score = distance[0][0] # should result a low distance. Replace one of IMG2 or IMG1 with IMG3 to see a high distance
    thres = 0.3146 #Threshold value obtained from running RunValidation.py on validation set
    if(distance_score < thres):
        print(f"Images are of the SAME person with distance threshold {distance_score}\n")
    else:
        print(f"Images are of DIFFERENT personswith distance threshold {distance_score}\n")