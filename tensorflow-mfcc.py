import tensorflow.compat.v1 as tf
from tensorflow.python.ops import gen_audio_ops as audio_ops
import sounddevice as sd
import numpy as np
import threading
import os

rec_duration = 0.020
sample_rate = 16000
num_channels = 1

def get_mfcc(waveform):
        # Run the spectrogram and MFCC ops to get a 2D audio: Short-time FFTs
        # background_clamp dims: [time, channels]
        sample_rate = 16000

        spectrogram = audio_ops.audio_spectrogram(
            waveform,
            window_size=320,
            stride=160)
        # spectrogram: [channels/batch, frames, fft_feature]

        # extract mfcc features from spectrogram by audio_ops.mfcc:
        # 1 Input is spectrogram frames.
        # 2 Weighted spectrogram into bands using a triangular mel filterbank
        # 3 Logarithmic scaling
        # 4 Discrete cosine transform (DCT), return lowest dct_coefficient_count
        mfccs = audio_ops.mfcc(
            spectrogram=spectrogram,
            sample_rate=sample_rate,
            upper_frequency_limit=7600,
            lower_frequency_limit=60,
            filterbank_channel_count=40,
            dct_coefficient_count=13)
        # mfcc: [channels/batch, frames, dct_coefficient_count]
        # remove channel dim
        mfccs = tf.squeeze(mfccs, axis=0)
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