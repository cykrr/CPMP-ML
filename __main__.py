#!/bin/python3
import os
import sys
import argparse
import pickle

from cpmp_ml import generate_model, generate_data, generate_data2
import numpy as np

def train_model(x, y, S, H, output, batch_size, epochs):
    import tensorflow as tf
    from tensorflow.keras import (layers, Input, Sequential, Model, optimizers)
    from tensorflow.keras.losses import BinaryCrossentropy

    # init tf
    device_name = tf.test.gpu_device_name()
    print("device_name", device_name)
    with tf.device(device_name):
      Fmodel=generate_model(S, H) # predice steps
      Fmodel.compile(
              loss=BinaryCrossentropy(),
              optimizer=optimizers.Adam(learning_rate=0.001),
              metrics=['mse']
        )

      Fmodel.fit(np.array(x_train), np.array(y_train), epochs=epochs,
                 batch_size=64, verbose=True) # TODO: batch_sz = sample_sz?
      print ("Dumping weights")
      Fmodel.save_weights(output)


# TODO: añadir opción de guardar la configuración con un nombre.
# TODO: opción de cargar la configuración al entrenar el modelo

# TODO: Exploración de datos: programa en el que se puedan revisar los layouts 
#       y sus rspectivas soluciones. (Cairo - ImGUI)?. Así como ver el 
#       desempeño del modelo en un caso particular.

# TODO: Interfaz gráfica que realice las mismas acciones que el script


# TODO: PAPER: Deep Learning Assisted Heuristic Tree Search for the Container 
#       Pre-marshalling Problem

# TODO: PAPER: NP-hard: van Brink and van der Zwaan - 2014


parser = argparse.ArgumentParser(
        prog = "CPMP-gen",
        description = "",
        epilog = ""
        )

# Problem size
parser.add_argument("-S", "--stacks", required = True)
parser.add_argument("-H", "--height", required = True)
parser.add_argument("-N", "--max-movements", required = True)
parser.add_argument("-ss", "--sample-size", required = True)

parser.add_argument("-e", "--epochs")

# What to do?
parser.add_argument("-tm", "--train-model", action = "store_true")
parser.add_argument("-tm2", "--train-model2", action = "store_true")
parser.add_argument("-gd", "--generate-data", action = "store_true") # .h5!
parser.add_argument("-gd2", "--generate-data2", action = "store_true") # .h5!

parser.add_argument("-i", "--input")
parser.add_argument("-o", "--output")

parser.add_argument("-v", "--verbose", action = "store_true")

args = parser.parse_args()

# Sanity checks

if not args.stacks:
    print("Error: -S [--stacks] is required")
    exit(1)
elif not args.stacks.isnumeric():
    print("Error: --stacks parameter (stack count) must be numerical")
    exit(1)

if not args.height:
    print("Error: -H [--height] is required")
    exit(1)
if not args.height.isnumeric():
    print("Error: --height parameter (stack height) must be numerical")
    exit(1)

if not args.max_movements:
    print("Error: -N [--max-movements] is required")
    exit(1)
if not args.max_movements.isnumeric():
    print("Error: -N [--max-movements] parameter must be numerical")
    exit(1)

if not args.sample_size:
    print("Error: -ss [--sample-size] is required")
    exit(1)
if not args.max_movements.isnumeric():
    print("Error:  -ss [--sample-size] parameter must be numerical")
    exit(1)

if (args.generate_data or args.generate_data2 or args.train_model) and not args.output:
    print("Error: no --output [-o] specified")
    exit(1)

if (args.train_model or args.generate_data2 or args.train_model2) and not args.input:
    print("Error: no --input [-i] specified")
    exit(1)

if (args.train_model or args.train_model2) and not args.epochs:
    print("Error: no --epochs [-e] specified")
    exit(1)


# Store variables

S = int(args.stacks)
H = int(args.height)
N = int(args.max_movements)
sample_size = int(args.sample_size)

output = args.output

if args.generate_data:
    x, y = generate_data(S, H, N, sample_size);
    # TODO: diff h5 & other formats?
    with open(output, "xb") as file:
        print ("Dumping data..")
        pickle.dump([x, y], file)

elif args.generate_data2:
    import tensorflow as tf
    from tensorflow.keras import (layers, Input, Sequential, Model, optimizers)
    from tensorflow.keras.losses import BinaryCrossentropy

    # init tf
    device_name = tf.test.gpu_device_name()
    print("device_name", device_name)
    with tf.device(device_name):
        model=generate_model(S, H) # predice steps
        model.compile(
              loss=BinaryCrossentropy(),
              optimizer=optimizers.Adam(learning_rate=0.001),
              metrics=['mse']
        )
        model.load_weights(args.input)
        x, y = generate_data2(model, S, H, N, sample_size, batch_size=64);

        with open(output, "xb") as file:
            print ("Dumping data..")
            pickle.dump([x, y], file)


elif args.train_model:
  with open(args.input, "rb") as file:
      print ("Loading data..")
      [x_train,y_train] = pickle.load(file)
      train_model(x_train, y_train, S, H, args.output,
                  int(args.sample_size), int(args.epochs))


exit (1)

if trained_model_file == None:
    print("Error: No input model path specified")
    sys.exit();

# print("Importing TensorFlow...")
def import_tf():
    import tensorflow as tf
    from tensorflow.keras import optimizers
    from tensorflow.keras.losses import BinaryCrossentropy
    return tf
tf = import_tf()

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



