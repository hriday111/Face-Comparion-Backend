import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras import layers, models, Model
from tensorflow.keras.saving import register_keras_serializable
import os
CSV = os.path.join('TrainingSet', 'lfw_pairs.csv')
BATCH_SIZE = 32
IMAGE_SHAPE = (105, 105, 3) 
VAL_CSV = os.path.join('TrainingSet', 'val.csv')


"""
This section of code is responsible for loading the image pairs and their labels from a CSV file,
processing the images, and creating a TensorFlow dataset suitable for training a Siamese network.
"""
def LoadNpyWrapper(path1, path2, label):
    def load_npy(p1, p2, lbl):
        img1 = np.load(p1.decode('utf-8'), allow_pickle=True)
        img2 = np.load(p2.decode('utf-8'), allow_pickle=True)
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

        return img1, img2, np.float32(lbl)
    
    return tf.numpy_function(load_npy, [path1, path2, label], [tf.float32, tf.float32, tf.float32])

def create_dataset(csv_file=CSV, batch_size=BATCH_SIZE):
        df = pd.read_csv(csv_file)

        dataset  = tf.data.Dataset.from_tensor_slices((df['img1_path'].values,
                                                       df['img2_path'].values,
                                                         df['label'].values))
        dataset = dataset.map(LoadNpyWrapper, num_parallel_calls=tf.data.AUTOTUNE)

        def set_shapes(img1, img2, label):
            img1.set_shape(IMAGE_SHAPE)
            img2.set_shape(IMAGE_SHAPE)
            label.set_shape([]) 
            return (img1, img2), label 
        
        dataset = dataset.map(set_shapes, num_parallel_calls=tf.data.AUTOTUNE)
        
        dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
        return dataset 

"""
A custom callback to stop training on low learning rates.
"""
class StopOnLowLR(tf.keras.callbacks.Callback):
    def __init__(self, min_lr=1e-5):
        super(StopOnLowLR, self).__init__()
        self.min_lr = min_lr

    def on_epoch_end(self, epoch, logs=None):
        # Get current learning rate
        current_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        
        # Check if it is too small
        if current_lr < self.min_lr:
            print(f"\n\n>>> STOPPING EARLY: Learning Rate ({current_lr:.2e}) dropped below threshold ({self.min_lr:.2e})\n")
            self.model.stop_training = True
"""
This section defines the Siamese network architecture using TensorFlow and Keras.
"""

@register_keras_serializable()
def l2_norm(x):
    return tf.math.l2_normalize(x, axis=1)

@register_keras_serializable()
def euclidean_distance(vects):
    x, y = vects
    sum_square = tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)
    return tf.sqrt(tf.maximum(sum_square, 1e-7))
def build_siamese_model(input_shape=IMAGE_SHAPE):
    input_layer = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (10,10), activation='relu')(input_layer)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, (7,7), activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, (4,4), activation='relu')(x)
    x = layers.MaxPooling2D()(x)

    x= layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4096, activation=None)(x)
    x = layers.Lambda(l2_norm, output_shape=(4096,))(x)
    embedding_model = Model(inputs=input_layer, outputs=x, name="embedding_net")
    inp_a = layers.Input(shape=input_shape, name="left_image")
    inp_b = layers.Input(shape=input_shape, name="right_image")

    vec_a = embedding_model(inp_a)
    vec_b = embedding_model(inp_b) 

    distance = layers.Lambda(euclidean_distance, output_shape=(1,))([vec_a, vec_b])

    model = Model(inputs=[inp_a, inp_b], outputs=distance)

    return model




def contrastive_loss(y_true, y_pred, m=1.0):
    p_squared = tf.square(y_pred)
    m_square = tf.square(tf.maximum(m - y_pred, 0))
    return tf.reduce_mean((1 - y_true) *m_square + y_true *  p_squared)
    
def inv_accuracy(y_true, y_pred):
    threshold = 0.5
    predictions = tf.cast(y_pred < threshold, tf.float32)
    return tf.reduce_mean(tf.cast(tf.equal(predictions, y_true), tf.float32))
callbacks = [
    # Save the model only when validation loss improves
    tf.keras.callbacks.ModelCheckpoint(
        filepath='best_siamese_model.keras',
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,       # Cut LR in half
        patience=3,       # Wait 3 epochs before cutting
        min_lr=0.00001
    ),
    StopOnLowLR(min_lr=2e-5),
]
if __name__ == "__main__":
    train_ds = create_dataset(CSV)
    model = build_siamese_model(IMAGE_SHAPE)

    model.compile(optimizer='adam', loss=contrastive_loss, metrics=[inv_accuracy])
    history = model.fit(
        train_ds,
        validation_data=create_dataset(VAL_CSV),
        epochs=30,
        callbacks=callbacks
        )
    model.save('siamese_model.keras')
