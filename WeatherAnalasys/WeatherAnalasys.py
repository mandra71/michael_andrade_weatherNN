## module for parsing CSV data sets
#from __future__ import print_function
#import csv
#import pandas as pd
#import tensorflow as tf
#import numpy as np
#import math
#from IPython import display
#from sklearn import metrics
#from matplotlib import cm
#from matplotlib import gridspec
#from matplotlib import pyplot as plt
#from tensorflow.examples.tutorials.mnist import input_data

import pandas as pd
import numpy as np
import tflearn
from tflearn.data_utils import load_csv, to_categorical
from decimal import Decimal, ROUND_HALF_UP
import matplotlib.pyplot as plt

pd.options.display.max_rows = 20
#pd.options.display.float_format = '{:.1f}'.format

# The file containing the weather samples
WEATHER_SAMPLE_FILE = 'weather.csv'

WeatherList = pd.read_csv(WEATHER_SAMPLE_FILE, sep=',')
#print(WeatherList.head())
#print(WeatherList.describe())

# Load CSV file, indicate that the first column represents labels
data, labels = load_csv(WEATHER_SAMPLE_FILE, columns_to_ignore=[1,10])

# Preprocessing functions
def preprocessData(data):
    for column in range(0,8):
        maximum = -10000000.0
        minimum = 10000000.0
        for findMaxMin in range(len(data)):
            maximum = float(data[findMaxMin][column]) if float(data[findMaxMin][column]) > maximum else maximum
            minimum = float(data[findMaxMin][column]) if float(data[findMaxMin][column]) < minimum else minimum
        for i in range(len(data)):
          data[i][column] = Decimal(Decimal((float(data[i][column]) - minimum) 
                                            / (maximum - minimum)).quantize(Decimal('.01'), rounding=ROUND_HALF_UP))
    return np.array(data, dtype=np.float32)

maximumEnergy = 0.0
for findMax in range(len(labels)):
    maximumEnergy = float(labels[findMax]) if float(labels[findMax]) > maximumEnergy else maximumEnergy
def preprocessLabelsNormalized(labels, maximumEnergy):
    for i in range(len(labels)):
        labels[i] = Decimal(Decimal(float(labels[i]) / maximumEnergy).quantize(Decimal('.01'), rounding=ROUND_HALF_UP))
    return np.array(labels, dtype=np.float32)
def preprocessLabels(labels):
    for i in range(len(labels)):
        next = float(labels[i])
        if next > 4000:
            labels[i] = 4
        elif next > 3000:
            labels[i] = 3
        elif next > 2000:
            labels[i] = 2
        elif next > 1000:
            labels[i] = 1
        else:
            labels[i] = 0
    return np.array(to_categorical(labels, 5), dtype=np.float32) 

# Preprocess data
data = preprocessData(data)
#labels = preprocessLabelsNormalized(labels, maximumEnergy)
labels = preprocessLabels(labels)
#labels = np.reshape(labels, (-1, 1))

# Build neural network
net = tflearn.input_data(shape=[None, 8])
net = tflearn.fully_connected(net, 32, weights_init='xavier', activation="softsign", name="layer1") 
net = tflearn.highway(net, 32, activation="softsign", name="layer2")
net = tflearn.fully_connected(net, 32, weights_init='xavier', activation="softsign", name="layer3")
net = tflearn.fully_connected(net, 5, activation='softmax', name="layer4")             
net = tflearn.regression(net, learning_rate=0.001)
model = tflearn.DNN(net, tensorboard_verbose=3, tensorboard_dir='./tmp/weather.log')
model.fit(data, labels, n_epoch=10, validation_set=0.01, show_metric=True, batch_size=6)

sample1 = data[22]
sample2 = data[6]
sampleLow = []
sampleHigh = []
for i in range(len(sample1)):
    sampleLow.append(sample1[i])
    sampleHigh.append(sample2[i])

pred = model.predict([sampleLow,sampleHigh])

print("Low solar energy: ")
print(pred[0].argmax())
print("High solar energy: ")
print(pred[1].argmax())

#print("Low solar energy: ", pred[0][0] * maximumEnergy)
#print("High solar energy: ", pred[1][0] * maximumEnergy)


#Plotting
#x = np.linspace(0, 10, 1)
#
##plt.plot([1,2,3,4,5,6,7,8,9,10], [.53,.68,.59,.58,.67,.58,.72,.61,.61,.63], label='Train Accuracy')
##plt.plot([1,2,3,4,5,6,7,8,9,10], [.50,.68,.62,.67,.68,.71,.68,.64,.71,.67], label='Validation Accuracy')
#plt.plot([1,2,3,4,5,6,7,8,9,10], [1.05,.78,.99,.98,.82,.98,.77,.94,.86,.84], label='Train Loss')
#plt.plot([1,2,3,4,5,6,7,8,9,10], [1.06,.89,.87,.85,.87,.83,.85,.87,.82,.84], label='Validation Loss')
#
#plt.xlabel('Epoch')
#plt.ylabel('Accuracy')
#
#plt.title("Accuracy vs. Epoch")
#
#plt.legend()
#
#plt.show()
