import tensorflow as tf
import sounddevice as sd1
import sounddevice as sd2
import sounddevice as sd3
import numpy as np
import threading

# Parameters
word_threshold = 7.5
word_duration = 10
rec_duration = 0.020
sample_rate = 16000
num_channels = 3

sd1.default.latency= ('high', 'high')
sd1.default.dtype= ('float32', 'float32')
sd1.default.device = "cap1 cap2 cap3"


# Load the TFLite model1 and allocate tensors.
interpreter1 = tf.lite.Interpreter(model_path="/home/pi/google-kws/models2/crnn_state-mffc-op/quantize_opt_for_size_tflite_stream_state_external/stream_state_external1.tflite")
interpreter1.allocate_tensors()
# Load the TFLite model2 and allocate tensors.
interpreter2 = tf.lite.Interpreter(model_path="/home/pi/google-kws/models2/crnn_state-mffc-op/quantize_opt_for_size_tflite_stream_state_external/stream_state_external2.tflite")
interpreter2.allocate_tensors()
# Load the TFLite model2 and allocate tensors.
interpreter3 = tf.lite.Interpreter(model_path="/home/pi/google-kws/models2/crnn_state-mffc-op/quantize_opt_for_size_tflite_stream_state_external/stream_state_external3.tflite")
interpreter3.allocate_tensors()
# Get input and output tensors1.
input_details1 = interpreter1.get_input_details()
output_details1 = interpreter1.get_output_details()
inputs1 = []
for s in range(len(input_details1)):
    inputs1.append(np.zeros(input_details1[s]['shape'], dtype=np.float32))
# Get input and output tensors2.
input_details2 = interpreter2.get_input_details()
output_details2 = interpreter2.get_output_details()
inputs2 = []
for s in range(len(input_details2)):
    inputs2.append(np.zeros(input_details2[s]['shape'], dtype=np.float32))
# Get input and output tensors3.
input_details3 = interpreter3.get_input_details()
output_details3 = interpreter3.get_output_details()
inputs3 = []
for s in range(len(input_details3)):
    inputs3.append(np.zeros(input_details3[s]['shape'], dtype=np.float32))
       
def get_prediction1(audio):
    # Make prediction from model
    interpreter1.set_tensor(input_details1[0]['index'], audio)
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
     

    print("p1 ", np.argmax(output_data[0]),output_data[0])

def get_prediction2(audio):
    # Make prediction from model
    interpreter2.set_tensor(input_details2[0]['index'], audio)
    # set input states (index 1...)
    for s in range(1, len(input_details2)):
      interpreter2.set_tensor(input_details2[s]['index'], inputs2[s])
  
    interpreter2.invoke()
    output_data = interpreter2.get_tensor(output_details2[0]['index'])
    # get output states and set it back to input states
    # which will be fed in the next inference cycle
    for s in range(1, len(input_details2)):
      # The function `get_tensor()` returns a copy of the tensor data.
      # Use `tensor()` in order to get a pointer to the tensor.
      inputs2[s] = interpreter2.get_tensor(output_details2[s]['index'])
     

    print("p2 ", np.argmax(output_data[0]),output_data[0])
    
def get_prediction3(audio):
    # Make prediction from model
    interpreter3.set_tensor(input_details3[0]['index'], audio)
    # set input states (index 1...)
    for s in range(1, len(input_details3)):
      interpreter3.set_tensor(input_details3[s]['index'], inputs3[s])
  
    interpreter3.invoke()
    output_data = interpreter3.get_tensor(output_details3[0]['index'])
    # get output states and set it back to input states
    # which will be fed in the next inference cycle
    for s in range(1, len(input_details3)):
      # The function `get_tensor()` returns a copy of the tensor data.
      # Use `tensor()` in order to get a pointer to the tensor.
      inputs3[s] = interpreter3.get_tensor(output_details3[s]['index'])
     

    #print("p2 ", np.argmax(output_data[0]),output_data[0])  
def sd_callback1(rec, frames, time, status):
    
    # Notify if errors
    if status:
        print('Error:', status)
    
    rec = np.reshape(rec, (1, 320))
    get_prediction1(rec)
    #get_prediction2(rec)
    #get_prediction3(rec)
    #print("p2 ", np.argmax(output_data[0]),output_data[0])  


# Start streaming from microphone
with sd1.InputStream(channels=num_channels,
                    samplerate=sample_rate,
                    blocksize=int(sample_rate * rec_duration),
                    callback=sd_callback1):
    threading.Event().wait()
