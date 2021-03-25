import sounddevice as sd
import sfeatpy
import numpy as np
import threading
import os

rec_duration = 0.020
sample_rate = 16000
num_channels = 1

def get_mfcc(audio):
   mfccs = sfeatpy.mfcc(np.squeeze(audio), sample_rate=16000, window_length=320, window_stride=160, max_freq=7600, min_freq=60, num_filter=40, num_coef=13, windowFun=0)
   return mfccs

def sd_callback(rec, frames, time, status):

   # Notify if errors
   if status:
       print('Error:', status)

   mfcc = get_mfcc(rec)
   #print(mfcc)
   print(os.getloadavg())
   
# Start streaming from microphone
with sd.InputStream(channels=num_channels,
                   samplerate=sample_rate,
                   blocksize=int(sample_rate * rec_duration),
                   callback=sd_callback):
    threading.Event().wait()