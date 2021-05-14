import tensorflow as tf
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import queue
import argparse

q = queue.Queue()

parser = argparse.ArgumentParser()
parser.add_argument(
            '-d', '--dump', type=str,
                help='dump recording to a file')
parser.add_argument(
            '-m', '--model', required=True, type=str,
                help='dump recording to a file')
args = parser.parse_args()


# Parameters
word_threshold = 7.5
word_duration = 10
rec_duration = 0.020
sample_rate = 16000
num_channels = 1

sd.default.latency= ('high', 'high')
sd.default.dtype= ('float32', 'float32')


# Load the TFLite model and allocate tensors.
interpreter1 = tf.lite.Interpreter(model_path=args.model)
interpreter1.allocate_tensors()

# Get input and output tensors.
input_details1 = interpreter1.get_input_details()
output_details1 = interpreter1.get_output_details()

inputs1 = []

for s in range(len(input_details1)):
  inputs1.append(np.zeros(input_details1[s]['shape'], dtype=np.float32))
    
kw_count = 0
not_kw_count = 0
kw_sum = 0
kw_hit = False
def sd_callback(rec, frames, time, status):
    if args.dump:
        q.put(rec.copy())
    global input_details1
    global output_details1
    global inputs1
    global kw_count
    global not_kw_count
    global kw_sum
    global kw_hit
    # Notify if errors
    if status:
        print('Error:', status)
    
    rec = np.reshape(rec, (1, 320))
    
    # Make prediction from model
    interpreter1.set_tensor(input_details1[0]['index'], rec)
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
     
    if np.argmax(output_data[0]) == 2:
      if kw_count > 3:
        print(output_data[0][2], kw_count, kw_sum)
        not_kw_count = 0
      kw_count += 1
      kw_sum = kw_sum + output_data[0][2]
      if kw_sum > 100:
        print("Kw threshold hit")
        kw_hit = True
    elif np.argmax(output_data[0]) != 2:
      if kw_hit:
        for s in range(len(input_details1)):
          inputs1[s] = np.zeros(input_details1[s]['shape'], dtype=np.float32)
        kw_hit = False
      if not_kw_count > 3:
        kw_count = 0
        kw_sum = 0
        not_kw_count = -1
      not_kw_count += 1

    
try:
  # Start streaming from microphone
  with sf.SoundFile(args.dump, mode='w', samplerate=sample_rate,
                                channels=num_channels) as file:
      with sd.InputStream(channels=num_channels,
                          samplerate=sample_rate,
                          blocksize=int(sample_rate * rec_duration),
                          callback=sd_callback):
          if args.dump:
              while True:
                  file.write(q.get())
          else:
              threading.Event().wait()

except KeyboardInterrupt:
    if args.dump:
      print('\nRecording dumped: ' + repr(args.dump))
    parser.exit('')
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))
