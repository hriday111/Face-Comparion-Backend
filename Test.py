import tensorflow as tf
import numpy as np
import cv2

MODEL = 'siamese_model.h5'
IMG1 = 'TrainingSet\lfw-deepfunneled\lfw-deepfunneled\Maria_Shriver\Maria_Shriver_0007.jpg'
IMG2 = 'TrainingSet\lfw-deepfunneled\lfw-deepfunneled\Maria_Shriver\Maria_Shriver_0002.jpg'


IMG3 = 'TrainingSet\lfw-deepfunneled\lfw-deepfunneled\Rodrigo_de_la_Cerna\Rodrigo_de_la_Cerna_0001.jpg'

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)



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


def euclidean_distance(vects):
    x, y = vects
    sum_square = tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)
    return tf.sqrt(tf.maximum(sum_square, 1e-7)) # 1e-7 used to avoid sqrt(0) and then division by zero
if __name__ == "__main__":
    model = tf.keras.models.load_model(MODEL,custom_objects={'euclidean_distance': euclidean_distance}, compile=False)
    img1 = preprocess_real_image('IMG1')
    img2 = preprocess_real_image('IMG2')
    distance = model.predict([img1, img2])
    distance_score = distance[0][0] # should result a low distance. Replace one of IMG2 or IMG1 with IMG3 to see a high distance
    
    print(f"Euclidean Distance between the two images: {distance_score}")