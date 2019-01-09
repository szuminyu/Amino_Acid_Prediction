import pandas as pd
import numpy as np

import string

# Maps aålphabets to numbers
di = dict(zip(string.ascii_letters, [ord(c) % 32 for c in string.ascii_letters]))

train_data = []

# Get the data in the dataframe
train_input = pd.read_csv('/Users/Sumin/Desktop/python/train_input.csv')

# LSTM inputs a ndarray and for the rows to be of the same column size, we look at the maximum column size
max_length = max(train_input['length'])

for index, row in train_input.iterrows():

    # If length of seq is 100 but the maximum length it can be is 200 then we know we need to add 200-100=100 more columns of zeros
    zeros_to_add = max_length - row['length']

    # For each element in sequence, save the corresponding number so['ABCD'] => ['1.0 2.0 3.0 4.0']
    temp_seq_list = list(row['sequence'])
    seq_list = []
    for i in temp_seq_list:
        seq_list.append(float(di[i]))

    # For each element in q8, save the corresponding number so['ABCD'] => ['1.0 2.0 3.0 4.0']
    temp_q8_list = list(row['sequence'])
    q8_list = []
    for i in temp_q8_list:
        q8_list.append(float(di[i]))

    # Here we multiply by 2(once for sequence and once for q8)
    temp_zero_list = [0] * (2 * zeros_to_add)

    # Just converting every element ot float
    zero_list = [float(i) for i in temp_zero_list]

    # This appends each row as sequence + q8 + number of zeros required
    train_data.append(seq_list + q8_list + temp_zero_list)




import numpy as np

# Just converting it to a numpy matrix
train_input = np.array(train_data)

train_input

# Load the labels matrix and calculate the average of each matrix
train_output = []
file = np.load('/Users/Sumin/Desktop/python/train_output.npz')
for key in file:
    train_output.append(np.average(file[key]))

# Convert the train_output to a numpy matrix
train_output = np.array(train_output)

# Displaying all the shapes
print("Train Input Shape: ", train_input.shape)
print("Train Output Shape: ", train_output.shape)




####### try to get the triangle from the matrix #######

####my code
train_output = []
file = np.load('/Users/Sumin/Desktop/python/train_output.npz')
for key in file:
    train_output.append(file[key])

train_output_a = []
for i in range(0, 4554):
    length = len(train_output[i])
    train_output_a.append(train_output[i][np.triu_indices(length, k=1)])

####River's

train_output = []
file = np.load('/Users/Sumin/Desktop/python/train_output.npz')
for key in file:
    train_output.append(file[key])

    train_output_a = []
    for i in train_output:
        train_output_a.append(i[np.triu_indices(len(i), k=1)])

####



train_output_a = np.array(train_output_a)
print(train_output_a.shape)

train_output_flatten = train_output_a.ravel()
train_output_a = np.ndarray.tolist(train_output_a)

# transfer train output matrix to array
train_output_array = []
#for i in train_output:
#    train_output_array.append(i[np.triu_indices(len(i), k=1)])
#
#train_output_array = np.array(train_output_array)

#print(train_output[0])



##Modeling

from tensorflow import keras
from keras.engine import Input

# Contructing LSTM having 100 neurons, using MSE as the loss function and Adam's optimizer
model = keras.Sequential()
model.add(keras.layers.LSTM(100, input_shape = (1,1382), return_sequnces = True))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dense(1,activation='relu'))
model.compile(loss='mean_squared_error', optimizer='adam')

# LSTM needs 3 input arguments so it will be (number of rows, timestamps, number of columns)
train_input = train_input.reshape((4554,1,1382))



model.fit(train_input,
                    train_output_a,
                    epochs=20,
                    validation_split=0.2,
                    verbose=2)



##experiment

model = keras.Sequential()
model.add(keras.layers.LSTM(100, input_shape = (1,1382)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dense(1,activation='relu'))
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(train_input, train_output_a, epochs = 10, validation_split = 0.2, verbose = 2)
