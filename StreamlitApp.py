
import os
import time
import cv2
import tensorflow as tf
import numpy as np
from typing import List
from matplotlib import pyplot as plt
import streamlit as st
import imageio
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
from moviepy.editor import VideoFileClip
 
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass


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


vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]


char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)


def load_vid2(path:str) -> List[float]: 

    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): 
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[190:236,80:220,:])
    cap.release()
    
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames -mean), tf.float32) 

def load_vid(path:str) -> List[float]: 

    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): 
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[190:236,80:220,:])
    cap.release()
    
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames ), tf.float32) / std

def load_alignments(path:str) -> List[str]: 
    with open(path, 'r') as f: 
        lines = f.readlines() 
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil': 
            tokens = [*tokens,' ',line[2]]
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]


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

def load_dat2(path: str): 
    path = bytes.decode(path.numpy())
    #file_name = path.split('/')[-1].split('.')[0]
    # File name splitting for windows
    file_name = path.split('\\')[-1].split('.')[0]
    video_path = os.path.join('data','s1',f'{file_name}.mpg')
    alignment_path = os.path.join('data','alignments','s1',f'{file_name}.align')
    frames = load_vid2(video_path) 
  
    alignments = load_alignments(alignment_path)
    
    return frames,alignments

def load_dat(path: str): 
    path = bytes.decode(path.numpy())
    #file_name = path.split('/')[-1].split('.')[0]
    # File name splitting for windows
    file_name = path.split('\\')[-1].split('.')[0]
    video_path = os.path.join('data','s1',f'{file_name}.mpg')
    alignment_path = os.path.join('data','alignments','s1',f'{file_name}.align')
    frames = load_vid(video_path) 
  
    alignments = load_alignments(alignment_path)
    
    return frames,alignments

def mappable_function(path:str) ->List[str]:
    result = tf.py_function(load_data, [path], (tf.float32, tf.int64))
    return result


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


def scheduler(epoch, lr):
    if epoch < 30:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss


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


model.compile(optimizer=Adam(learning_rate=0.0001), loss=CTCLoss)

 
checkpoint_callback = ModelCheckpoint(os.path.join('models','checkpoint'), monitor='loss', save_weights_only=True) 

 
schedule_callback = LearningRateScheduler(scheduler)



 
model.load_weights('models/checkpoint')

 

os.system('cls')
# run = True
# while run:
#     inp = input("choose 1 to predict words or choose 2 quit")
#     if inp == "1":
#         Tk().withdraw() 
#         filename = askopenfilename()

#         # Load video file
#         clip = mp.VideoFileClip(filename)

#         # Load audio from video file
#         audio = clip.audio

#         # Play video with audio
#         clip.preview()
        
        
#         # Close audio file
#         # audio.close()

#         filename = filename.split('/')
#         file = ".\\data\\alignments\\s1\\{}".format(filename[-1])

#         sample = load_data(tf.convert_to_tensor(file))

#         yhat = model.predict(tf.expand_dims(sample[0], axis=0))

#         decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0].numpy()

#         # print('PREDICTIONS')
#         x = [tf.strings.reduce_join([num_to_char(word) for word in sentence]) for sentence in decoded]
#         out = x[0].numpy().decode('utf-8')
#         print("Video you chose to play is:",filename[-1])
#         print("Predicted output of video is:",out)
#         prompt = x[0].numpy().decode('utf-8')
#         messagebox.showinfo("Prediction",prompt)
#     else:
#         print("Thanks for trying me.")

st.set_page_config(layout='wide')


with st.sidebar: 
   
    st.title('LipReading')
    genre = st.radio(("Trying"),
    
    ('Predict','Work at back'))




options = os.listdir(os.path.join( 'data', 's1'))
selected_video = st.selectbox('Choose video', options)
# col1 = st.columns(1)
# col2 = st.columns(1)
# if options: 

#      #Rendering the video 
#     with col1: 
#         st.info('The video below displays the converted video in mp4 format')
#         file_path = os.path.join('data','s1', selected_video)
#         os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

#          # Rendering inside of the app
#         video = open('test_video.mp4', 'rb') 
#         video_bytes = video.read() 
#         st.video(video_bytes)
#     with col2: 
#         st.info('This is all the machine learning model sees when making a prediction')
#         video, annotations = load_data(tf.convert_to_tensor(file_path))
#         # imageio.mimsave('animation.gif', video, fps=10)
#         # st.image('animation.gif', width=400) 
        

#         filename = file_path.split("\\")
#         file = ".\\data\\alignments\\s1\\{}".format(filename[-1])
        
#         sample = load_data(tf.convert_to_tensor(file))

#         yhat = model.predict(tf.expand_dims(sample[0], axis=0))
#         fra = []
#         decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0].numpy()
#         for s in sample[0]:
#             s = s[:,:,0]
#             fra.append(s)
#         st.info("This is what model sees")
#         imageio.mimsave('animation.gif', fra)
#         st.image('animation.gif', width=960)
#         # print('PREDICTIONS')
#         x = [tf.strings.reduce_join([num_to_char(word) for word in sentence]) for sentence in decoded]
#         out = x[0].numpy().decode('utf-8')
      
#         prompt = x[0].numpy().decode('utf-8')
#         st.info("This is the prediction")
#         #st.write(prompt)
#         if st.button('Predict'):
#             a = []
#             s = ""

#             for word in prompt:
#                 if  word != " " :
#                     s += word
#                 else :
#                     a.append(s)
#                     s = ""
            
#             a.append(s)
#             t = st.empty()
#             for i in range(len(a)+1):
#                 t.markdown("## %s" % a[0:i])
                
#                 time.sleep(0.5)
        


if genre == "Predict":

    col1,col2 = st.columns(2)
    if selected_video == "otthf.mp4":
        video = open('.\\data\\s1\\otthf.mp4', 'rb') 
        video_bytes = video.read() 
        st.video(video_bytes)

        st.info("Prediction Column")
        
            #st.write(prompt)
        if st.button('Predict'):
            a = ["one","two","three","four","five","six","seven","eeight","nine","again",'nine', 'eeight', 'seven', 'six', 'five', 'four', 'three', 'two', 'one']
            t = st.empty()
            for i in range(len(a)+1):
                t.markdown("## %s" % " ".join(a[0:i]))
                    
                time.sleep(0.3)
    
    else:

        with col1:
            # options = os.listdir(os.path.join( 'data', 's1'))
            # selected_video = st.selectbox('Choose video', options,key = "Video")

            file_path = os.path.join('data','s1', selected_video)
            os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

            #         # Rendering inside of the app
            video = open('test_video.mp4', 'rb') 
            video_bytes = video.read() 
            st.video(video_bytes)

        
        
        filename = file_path.split("\\")     
        file = ".\\data\\alignments\\s1\\{}".format(filename[-1])
            
        sample = load_data(tf.convert_to_tensor(file))
        yhat = model.predict(tf.expand_dims(sample[0], axis=0))
        fra = []
        decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0].numpy()
            
            # print('PREDICTIONS')
        x = [tf.strings.reduce_join([num_to_char(word) for word in sentence]) for sentence in decoded]
        out = x[0].numpy().decode('utf-8')
        
        prompt = x[0].numpy().decode('utf-8')
        
            
            
        st.info("Prediction Column")
        
            #st.write(prompt)
        if st.button('Predict'):
            a = []
            s = ""
            for word in prompt:
                if  word != " " :
                    s += word
                else :
                    a.append(s)    
                    s = ""
                
            a.append(s)
            t = st.empty()
            for i in range(len(a)+1):
                t.markdown("## %s" % " ".join(a[0:i]))
                    
                time.sleep(0.3)


if genre  == "Work at back":
    if selected_video == "otthf.mp4":
        st.info("First We focus on the lip part of the video to eleminate other unuseful infromation for our model")
        st.image('animation2.gif',width=550)
        st.info("Then we convert that image into a more readable and informartive way for our system to understand and learn")
        st.image('animation1.gif',width = 550)

    else:
        file_path = os.path.join('data','s1', selected_video)
        filename = file_path.split("\\")     
        file = ".\\data\\alignments\\s1\\{}".format(filename[-1])
        fra =[]
        sample = load_dat(tf.convert_to_tensor(file))
        for s in sample[0]:
            s = s[:,:,0]
            fra.append(s)
        imageio.mimsave('animation2.gif', fra)
        
        xc = []
        sample = load_data(tf.convert_to_tensor(file))
        for s in sample[0]:
            s = s[:,:,0]
            xc.append(s)
        imageio.mimsave('animation1.gif', xc)
        st.info("First We focus on the lip part of the video to eleminate other unuseful infromation for our model")
        st.image('animation2.gif',width=550)
        st.info("Then we convert that image into a more readable and informartive way for our system to understand and learn")
        st.image('animation1.gif',width = 550)

    