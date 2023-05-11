# %% [markdown]
# # 0. Install and Import Dependencies

# %%
import os
import cv2
import tensorflow as tf
import numpy as np
from typing import List
from matplotlib import pyplot as plt
import streamlit

from tkinter import Tk    
from tkinter.filedialog import askopenfilename
import moviepy.editor as mp
from moviepy import *
from tkinter import Tk,messagebox    
from tkinter.filedialog import askopenfilename
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

# %%
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

#print('REAL TEXT')
# [tf.strings.reduce_join([num_to_char(word) for word in sentence]) for sentence in [sample[1]]]

# %% [markdown]
# # 1. Build Data Loading Functions

# %%
def load_video(path:str) -> List[float]: 

    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): 
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[190:236,80:220,:])
    cap.release()
    
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std

# %%
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]

# %%
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)



# %%
def load_alignments(path:str) -> List[str]: 
    with open(path, 'r') as f: 
        lines = f.readlines() 
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil': 
            tokens = [*tokens,' ',line[2]]
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]

# %%
def load_data(path: str): 
    path = bytes.decode(path.numpy())
    #file_name = path.split('/')[-1].split('.')[0]
    # File name splitting for windows
    file_name = path.split('\\')[-1].split('.')[0]
    video_path = os.path.join('data','s1',f'{file_name}.mpg')
    alignment_path = os.path.join('data','alignments','s1',f'{file_name}.align')
    frames = load_video(video_path) 
    alignments = load_alignments(alignment_path)
    
    return frames,alignments

# %%
def mappable_function(path:str) ->List[str]:
    result = tf.py_function(load_data, [path], (tf.float32, tf.int64))
    return result

# %% [markdown]
# # 2. Create Data Pipeline

# %% [markdown]
# # 3. Design the Deep Neural Network

# %%


# %%
model = Sequential()
model.add(Conv3D(128, 3, input_shape=(75,46,140,1), padding='same'))
model.add(Activation('relu'))
model.add(MaxPool3D((1,2,2)))

model.add(Conv3D(256, 3, padding='same'))
model.add(Activation('relu'))
model.add(MaxPool3D((1,2,2)))

model.add(Conv3D(75, 3, padding='same'))
model.add(Activation('relu'))
model.add(MaxPool3D((1,2,2)))

model.add(TimeDistributed(Flatten()))

model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
model.add(Dropout(.5))

model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
model.add(Dropout(.5))

model.add(Dense(char_to_num.vocabulary_size()+1, kernel_initializer='he_normal', activation='softmax'))

# %% [markdown]
# # 4. Setup Training Options and Train

# %%
def scheduler(epoch, lr):
    if epoch < 30:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

# %%
def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

# %%
class ProduceExample(tf.keras.callbacks.Callback): 
    def __init__(self, dataset) -> None: 
        self.dataset = dataset.as_numpy_iterator()
    
    def on_epoch_end(self, epoch, logs=None) -> None:
        data = self.dataset.next()
        yhat = self.model.predict(data[0])
        decoded = tf.keras.backend.ctc_decode(yhat, [75,75], greedy=False)[0][0].numpy()
        for x in range(len(yhat)):           
            print('Original:', tf.strings.reduce_join(num_to_char(data[1][x])).numpy().decode('utf-8'))
            print('Prediction:', tf.strings.reduce_join(num_to_char(decoded[x])).numpy().decode('utf-8'))
            print('~'*100)

# %%
model.compile(optimizer=Adam(learning_rate=0.0001), loss=CTCLoss)

# %%
checkpoint_callback = ModelCheckpoint(os.path.join('models','checkpoint'), monitor='loss', save_weights_only=True) 

# %%
schedule_callback = LearningRateScheduler(scheduler)

# %% [markdown]
# # 5. Make a Prediction 

# %%
model.load_weights('models/checkpoint')

# %%

os.system('cls')
run = True
while run:
    inp = input("choose 1 to predict words or choose 2 quit")
    if inp == "1":
        Tk().withdraw() 
        filename = askopenfilename()

        # Load video file
        clip = mp.VideoFileClip(filename)

        # Load audio from video file
        audio = clip.audio

        # Play video with audio
        clip.preview()
        
        
        # Close audio file
        # audio.close()

        filename = filename.split('/')
        file = ".\\data\\alignments\\s1\\{}".format(filename[-1])

        sample = load_data(tf.convert_to_tensor(file))

        yhat = model.predict(tf.expand_dims(sample[0], axis=0))

        decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0].numpy()

        # print('PREDICTIONS')
        x = [tf.strings.reduce_join([num_to_char(word) for word in sentence]) for sentence in decoded]
        out = x[0].numpy().decode('utf-8')
        print("Video you chose to play is:",filename[-1])
        print("Predicted output of video is:",out)
        prompt = x[0].numpy().decode('utf-8')
        messagebox.showinfo("Prediction",prompt)
    else:
        print("Thanks for trying me.")

# %% [markdown]
# # Test on a Video

# %%
# Tk().withdraw() 
# filename = askopenfilename()
# filename = filename.split('/')
# file = ".\\data\\alignments\\s1\\{}".format(filename[-1])
# #sample = load_data(tf.convert_to_tensor('.\\data\\alignments\\s1\\bbaf2n.mpg'))
# sample = load_data(tf.convert_to_tensor(file))

# %% [markdown]
# 

# %%
# yhat = model.predict(tf.expand_dims(sample[0], axis=0))

# %%
# decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0].numpy()

# %%
# print('PREDICTIONS')
# [tf.strings.reduce_join([num_to_char(word) for word in sentence]) for sentence in decoded]

# %%



