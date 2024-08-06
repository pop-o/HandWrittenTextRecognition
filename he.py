import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Path to the IAM dataset
DATA_DIR = 'path/to/iamdataset/'

# Constants
IMG_WIDTH = 128
IMG_HEIGHT = 32
BATCH_SIZE = 32
MAX_TEXT_LEN = 32

# Function to load and preprocess images and labels
def preprocess_image_label(file_path, label):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype(np.float32) / 255.0
    return img, label

# Function to load IAM dataset
def load_iam_dataset(data_dir):
    images = []
    labels = []
    with open(os.path.join(data_dir, 'ascii/words.txt'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            if not line.startswith('#'):
                parts = line.strip().split()
                img_path = os.path.join(data_dir, 'words', parts[0].replace('-', '/')) + '.png'
                label = parts[-1]
                if os.path.exists(img_path):
                    img, lbl = preprocess_image_label(img_path, label)
                    images.append(img)
                    labels.append(lbl)
    return np.array(images), labels

# Load dataset
images, labels = load_iam_dataset(DATA_DIR)

# Encode labels
char_list = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
char_to_num = {char: i for i, char in enumerate(char_list)}
num_to_char = {i: char for i, char in enumerate(char_list)}

def encode_label(text):
    return [char_to_num[char] for char in text]

encoded_labels = [encode_label(label) for label in labels]
encoded_labels = pad_sequences(encoded_labels, maxlen=MAX_TEXT_LEN, padding='post')

# Train-test split
x_train, x_val, y_train, y_val = train_test_split(images, encoded_labels, test_size=0.2, random_state=42)

# Build the model
def build_model(img_width, img_height, max_text_len, char_list):
    input_img = layers.Input(shape=(img_height, img_width, 1), name='image')
    labels = layers.Input(name='label', shape=[max_text_len], dtype='float32')
    
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)
    x = layers.Reshape(target_shape=((img_width // 8), (img_height // 8) * 128))(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dense(len(char_list) + 1, activation='softmax')(x)
    
    y_pred = layers.Lambda(lambda x: x[:, 2:, :])(x)
    
    model = Model(inputs=input_img, outputs=y_pred)
    
    def ctc_lambda_func(args):
        y_pred, labels, input_length, label_length = args
        return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)
    
    input_length = layers.Input(name='input_length', shape=[1], dtype='int64')
    label_length = layers.Input(name='label_length', shape=[1], dtype='int64')
    ctc_loss = layers.Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    
    model_ctc = Model(inputs=[input_img, labels, input_length, label_length], outputs=ctc_loss)
    return model, model_ctc

model, model_ctc = build_model(IMG_WIDTH, IMG_HEIGHT, MAX_TEXT_LEN, char_list)

# Compile the model
model_ctc.compile(optimizer='adam')

# Data generator
def data_generator(x, y, batch_size):
    while True:
        for i in range(0, len(x), batch_size):
            x_batch = x[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            input_length = np.ones((len(x_batch), 1)) * (IMG_WIDTH // 8 - 2)
            label_length = np.ones((len(y_batch), 1)) * MAX_TEXT_LEN
            inputs = {
                'image': np.expand_dims(x_batch, -1),
                'label': y_batch,
                'input_length': input_length,
                'label_length': label_length,
            }
            outputs = {'ctc': np.zeros((len(x_batch),))}
            yield inputs, outputs

train_gen = data_generator(x_train, y_train, BATCH_SIZE)
val_gen = data_generator(x_val, y_val, BATCH_SIZE)

# Train the model
model_ctc.fit(train_gen,
              steps_per_epoch=len(x_train) // BATCH_SIZE,
              validation_data=val_gen,
              validation_steps=len(x_val) // BATCH_SIZE,
              epochs=50)

# Save the model
model.save('handwritten_text_recognizer.h5')
