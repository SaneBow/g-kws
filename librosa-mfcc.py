import sounddevice as sd
import numpy as np
import librosa
import threading
import os

rec_duration = 0.020
sample_rate = 16000
num_channels = 1
sd.default.never_drop_input= False
sd.default.latency= ('high', 'high')
sd.default.dtype= ('float32', 'float32')



def get_mfcc(audio):
    audio = np.squeeze(audio, 1)    
    mfccs = librosa.feature.mfcc(y=audio, n_mfcc=13, sr=sample_rate, n_fft=320, hop_length=160, n_mels=40, fmin=60.0, fmax=76000.0, htk=False )
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
