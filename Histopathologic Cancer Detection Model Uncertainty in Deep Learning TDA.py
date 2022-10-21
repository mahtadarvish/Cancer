import keras
from keras import backend as K
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

import kmapper as km
import numpy as np
import pandas as pd
from sklearn import metrics, cluster, preprocessing
import xgboost as xgb

from matplotlib import pyplot as plt
plt.style.use("ggplot")

############################################################################
#Preparing Data#############################################################
############################################################################

# # get the data, shuffled and split between train and test sets
dataset = pd.read_csv('TrainImgsBnW - With Label.csv')
x = dataset.iloc[:,3:].values
y = dataset.iloc[:,1].values

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

# alpha = np.append(X_test, dataset.iloc[0,3:].values)
# beta = np.append(Y_test, dataset.iloc[0,1])

alpha = np.append(dataset.iloc[0,3:].values, X_test)
beta = np.append(dataset.iloc[0,1], Y_test)

X_test = np.reshape(alpha, (X_test.shape[0]+1, X_test.shape[1]))
Y_test = beta

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

y_train = Y_train
y_test = Y_test

y_mean_test = y_test.mean()
print(y_mean_test, 'y test mean')

############################################################################
#Model######################################################################
############################################################################

batch_size = 128
num_classes = 1
epochs = 3

model = Sequential()
model.add(Dropout(0.5, input_shape=(576,)))
model.add(Dense(512, activation='relu', input_shape=(576,)))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='sigmoid'))

model.summary()
model.compile(loss='binary_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

############################################################################
#Fitting and evaluation#####################################################
############################################################################

history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)
# score = model.evaluate(X_test, y_test, verbose=0)
# # print(score)
# print('Test Loss:', score[0])
# print('Test accuracy:', score[1])

############################################################################
#Perform 1000 forward passes on test set and calculate Variance Ratio and Standard Dev
############################################################################

FP = 1000
predict_stochastic = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])

y_pred_test = np.array([predict_stochastic([X_test, 1]) for _ in range(FP)])
y_pred_stochastic_test = y_pred_test.reshape(-1,X_test.shape[0]).T

y_pred_std_test = np.std(y_pred_stochastic_test, axis=1)
y_pred_mean_test = np.mean(y_pred_stochastic_test, axis=1)
y_pred_mode_test = (np.mean(y_pred_stochastic_test > .5, axis=1) > .5).astype(int).reshape(-1,1)

y_pred_var_ratio_test = 1 - np.mean((y_pred_stochastic_test > .5) == y_pred_mode_test, axis=1)

test_analysis = pd.DataFrame({
    # "y_true": y_test,
    "y_pred": y_pred_mean_test,
    "VR": y_pred_var_ratio_test,
    "STD": y_pred_std_test
})

# print(metrics.accuracy_score(y_true=y_test, y_pred=y_pred_mean_test > .5))
# print(test_analysis.describe())

############################################################################
############################################################################
#Apply MAPPER###############################################################
############################################################################
############################################################################

############################################################################
#Take penultimate layer activations from test set for the inverse X#########
############################################################################

predict_penultimate_layer =  K.function([model.layers[0].input, K.learning_phase()], [model.layers[-2].output])

X_inverse_test = np.array(predict_penultimate_layer([X_test, 1]))[0]
print((X_inverse_test.shape, "X_inverse_test shape"))

############################################################################
#Take STD and error as the projected  X#####################################
############################################################################

# X_projected_test = np.c_[test_analysis.STD, test_analysis.y_true - test_analysis.y_pred]
X_projected_test = np.c_[test_analysis.STD, test_analysis.y_pred]
print((X_projected_test.shape, "X_projected_test shape"))

############################################################################
#Create the confidence graph  G#############################################
############################################################################

mapper = km.KeplerMapper(verbose=2)
G = mapper.map(X_projected_test,
               X_inverse_test,
               clusterer=cluster.AgglomerativeClustering(n_clusters=2),
               overlap_perc=0.2,
               nr_cubes=30)

############################################################################
#Create color function output (absolute error)##############################
############################################################################

# color_function_output = np.sqrt((y_test-test_analysis.y_pred)**2)
color_function_output = np.sqrt((test_analysis.y_pred)**2)

############################################################################
#Visualize##################################################################
############################################################################

_ = mapper.visualize(G,
                     lens=X_projected_test,
                     lens_names=["Uncertainty", "Error"],
                     color_function=color_function_output.values,
                     title="Confidence Graph for a MLP trained on Histopathologic Cancer",
                     path_html="confidence_graph_output.html")