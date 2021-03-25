import tensorflow.compat.v1 as tf
from tensorflow.python.ops import gen_audio_ops as audio_ops
import sounddevice as sd
import numpy as np
import threading
import os

rec_duration = 0.020
sample_rate = 16000
num_channels = 1

# Load the TFLite model and allocate tensors.
interpreter1 = tf.lite.Interpreter(model_path="/home/pi/google-kws/models2/crnn_state/quantize_opt_for_size_tflite_stream_state_external/stream_state_external.tflite")
interpreter1.allocate_tensors()

# Get input and output tensors.
input_details1 = interpreter1.get_input_details()
output_details1 = interpreter1.get_output_details()
inputs1 = []

for s in range(len(input_details1)):
  print(input_details1[s]['shape'])
  inputs1.append(np.zeros(input_details1[s]['shape'], dtype=np.float32))


# Load the TFLite model and allocate tensors.
interpreter2 = tf.lite.Interpreter(model_path="/home/pi/google-kws/models2/crnn_state/quantize_opt_for_size_tflite_stream_state_external/stream_state_external.tflite")
interpreter2.allocate_tensors()

# Get input and output tensors.
input_details2 = interpreter1.get_input_details()
output_details2 = interpreter1.get_output_details()
inputs2 = []

for s in range(len(input_details2)):
  print(input_details2[s]['shape'])
  inputs2.append(np.zeros(input_details2[s]['shape'], dtype=np.float32))

def get_prediction1(mfcc):
  
  # Make prediction from model
  interpreter1.set_tensor(input_details1[0]['index'], mfcc)
  # set input states (index 1...)
  for s in range(1, len(input_details1)):
      interpreter1.set_tensor(input_details1[s]['index'], inputs1[s])
  
  interpreter1.invoke()
  output_data = interpreter1.get_tensor(output_details1[0]['index'])
  # get output states and set it back to input states
  # which will be fed in the next inference cycle
  for s in range(1, len(input_details1)):
    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    inputs1[s] = interpreter1.get_tensor(output_details1[s]['index'])
     

  #print(np.argmax(output_data[0]),output_data[0])

def get_prediction2(mfcc):
  
  # Make prediction from model
  interpreter2.set_tensor(input_details2[0]['index'], mfcc)
  # set input states (index 1...)
  for s in range(1, len(input_details2)):
      interpreter2.set_tensor(input_details2[s]['index'], inputs1[s])
  
  interpreter2.invoke()
  output_data = interpreter2.get_tensor(output_details2[0]['index'])
  # get output states and set it back to input states
  # which will be fed in the next inference cycle
  for s in range(1, len(input_details2)):
    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    inputs2[s] = interpreter2.get_tensor(output_details2[s]['index'])
     

  #print(np.argmax(output_data[0]),output_data[0])



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
            dct_coefficient_count=20)
        # mfcc: [channels/batch, frames, dct_coefficient_count]
        # remove channel dim
        
        return mfccs



def sd_callback(rec, frames, time, status):
    
    # Notify if errors
    if status:
        print('Error:', status)
    mfcc = get_mfcc(rec)
    get_prediction1(mfcc)
    get_prediction2(mfcc)
    #print(os.getloadavg())
    #sd.wait()
# Start streaming from microphone
with sd.InputStream(channels=num_channels,
                    samplerate=sample_rate,
                    blocksize=int(sample_rate * rec_duration),
                    callback=sd_callback):
    threading.Event().wait()