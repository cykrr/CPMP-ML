import os
import sys
import argparse
import pickle

parser = argparse.ArgumentParser(
        prog = "CPMP-v3",
        description = "",
        epilog = ""
        )

parser.add_argument("-id", "--input-data")
parser.add_argument("-im", "--input-model")
parser.add_argument("-v", "--verbose", action = "store_true" )
args = parser.parse_args()
greedy_training_data_file = args.input_data
trained_model_file = args.input_model


if greedy_training_data_file == None:
    print("Error No input data path specified")
    sys.exit();

if trained_model_file == None:
    print("Error No input model path specified")
    sys.exit();

print("Importing TensorFlow...")
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.losses import BinaryCrossentropy
from cpmp_ml import generate_model
from cpmp_ml import generate_data
import numpy as np

S = ""
H = ""
N = ""

# gen train data if not recv'd


## If input file doesn't exist
if os.path.exists(greedy_training_data_file) == False:
  print("no training data, generating...")
  x_train, y_train = generate_data(sample_size=64,
                                   S=5, H=5, N=15,
                                   perms_by_layout=25, verbose=args.verbose)
# save train generated
  with open(greedy_training_data_file, "xb") as file:
      print ("Dumping data..")
      pickle.dump([x_train,y_train], file)
## if input file does exist
else:
  with open(greedy_training_data_file, "rb") as file:
      print ("Loading data..")
      [x_train,y_train] = pickle.load(file)
    
# init tf
device_name = tf.test.gpu_device_name()
print("device_name", device_name)
with tf.device(device_name):
  Fmodel=generate_model() # predice steps
  Fmodel.compile(
          loss=BinaryCrossentropy(),
          optimizer=optimizers.Adam(learning_rate=0.001),
          metrics=['mse']
    )


## If input file doesn't exist
if os.path.exists(trained_model_file) == False and os.path.exists(trained_model_file + ".index") == False:
  print("input model doesn't exist, training...")
  ## Training
  Fmodel.fit(np.array(x_train), np.array(y_train),
             epochs=1, verbose=True, batch_size=64)
  Fmodel.save_weights(trained_model_file);
else:
  Fmodel.load_weights(trained_model_file)



