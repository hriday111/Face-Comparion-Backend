import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, Model
from tensorflow.keras.saving import register_keras_serializable
import cv2
MODEL = 'siamese_model.keras'
RAW_PAIRS_CSV = 'ValidationSet/RawFaces/lfw_pairs1.csv'

@register_keras_serializable()
def l2_norm(x):
    return tf.math.l2_normalize(x, axis=1)

@register_keras_serializable()
def euclidean_distance(vects):
    x, y = vects
    sum_square = tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)
    return tf.sqrt(tf.maximum(sum_square, 1e-7))
model = tf.keras.models.load_model(
    MODEL,
    custom_objects={'euclidean_distance': euclidean_distance},
    compile=False,
    safe_mode=False
)
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
)

df = pd.read_csv(RAW_PAIRS_CSV)

def preprocess_image(path):
    img = cv2.imread(path)
    img_cropped = img[60:190, 60:190]  # Crop to focus on face area
    img_resized = cv2.resize(img_cropped, (105, 105))
    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_normalized = np.expand_dims(img_normalized, axis=0)
    return img_normalized

def preprocess_real_image(image_path):
    img = cv2.imread(image_path)
    faces = face_classifier.detectMultiScale(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=7, minSize=(30, 30)
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

print(f"Running predictions on {len(df)} pairs...")
distances = []
y_true = []
ValueErrorCount = 0
for index, row in df.iterrows():
    #print(row['img1_path'], row['img2_path'])
    try:
        img1 = preprocess_real_image(row['img1_path'])
        img2 = preprocess_real_image(row['img2_path'])  
    except ValueError as e:
        ValueErrorCount += 1
        print(f"Skipping pair at index {index} due to error: {e}")
        continue
    pred = model.predict([img1, img2], verbose=0)
    dist = pred[0][0]
    distances.append(dist)
    y_true.append(row['label'])

distances = np.array(distances)
y_true = np.array(y_true)

fpr, tpr, thresholds = roc_curve(y_true, -distances) # -distance due to inverse proportionality
eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
thresh = interp1d(fpr, thresholds)(eer)
print(f"Total errors due to face detection issues: {ValueErrorCount}")
print("\n--- METRICS REPORT ---")
print(f"Equal Error Rate (EER): {eer:.4f} ({(eer*100):.2f}%)")
print(f"Threshold at EER:       {-thresh:.4f}")

best_acc = 0
best_thresh = 0 
for t in np.linspace(distances.min(), distances.max(), num=1000):
    y_pred = (distances < t).astype(int)
    acc = accuracy_score(y_true, y_pred)
    
    if acc > best_acc:
        best_acc = acc
        best_thresh = t
print(f"Max Accuracy:           {best_acc:.4f} ({(best_acc*100):.2f}%)")
print(f"Best Accuracy Threshold:{best_thresh:.4f}")

plt.figure()
plt.plot(fpr, tpr, label=f'ROC (EER={eer:.2f})')
plt.plot([0, 1], [0, 1], 'k--') # Diagonal random guess line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()