# Imports
import numpy as np 
import pandas as pd 
from glob import glob 
from skimage.io import imread 
import os
import shutil
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.nasnet import NASNetMobile
from keras.applications.xception import Xception
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Input, Concatenate, GlobalMaxPooling2D
from keras.models import Model
from keras.callbacks import CSVLogger, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
from livelossplot import PlotLossesKeras

print('step 0')
# Output files
TRAINING_LOGS_FILE = "training_logs.csv"
MODEL_SUMMARY_FILE = "model_summary.txt"
MODEL_FILE = "histopathologic_cancer_detector.h5"
TRAINING_PLOT_FILE = "training.png"
VALIDATION_PLOT_FILE = "validation.png"
ROC_PLOT_FILE = "roc.png"
KAGGLE_SUBMISSION_FILE = "kaggle_submission.csv"
INPUT_DIR = 'Extra Data/'


# Hyperparams
SAMPLE_COUNT = 85000
TRAINING_RATIO = 0.9
IMAGE_SIZE = 96
EPOCHS = 2
BATCH_SIZE = 216
VERBOSITY = 1
TESTING_BATCH_SIZE = 5000


# # Data setup
# print('step 1')
# training_dir = INPUT_DIR + 'train/'
# data_frame = pd.DataFrame({'path': glob(os.path.join(training_dir,'*.tif'))})
# data_frame['id'] = data_frame.path.map(lambda x: x.split('/')[1].split('\\')[1].split('.')[0]) 
# labels = pd.read_csv(INPUT_DIR + 'train_labels.csv')
# data_frame = data_frame.merge(labels, on = 'id')
# negatives = data_frame[data_frame.label == 0].sample(SAMPLE_COUNT)
# positives = data_frame[data_frame.label == 1].sample(SAMPLE_COUNT)
# data_frame = pd.concat([negatives, positives]).reset_index()
# data_frame = data_frame[['path', 'id', 'label']]
# data_frame['image'] = data_frame['path'].map(imread)

# print('step 2')
training_path = 'Extra Data/training'
validation_path = 'Extra Data/validation'

# for folder in [training_path, validation_path]:
#     for subfolder in ['0', '1']:
#         path = os.path.join(folder, subfolder)
#         os.makedirs(path, exist_ok=True)

# training, validation = train_test_split(data_frame, train_size=TRAINING_RATIO, stratify=data_frame['label'])

# data_frame.set_index('id', inplace=True)

# print('step 3')
# for images_and_path in [(training, training_path), (validation, validation_path)]:
# 	images = images_and_path[0]
# 	path = images_and_path[1]
# 	for image in images['id'].values:
# 		file_name = image + '.tif'
# 		label = str(data_frame.loc[image,'label'])
# 		destination = os.path.join(path, label, file_name)
# 		if not os.path.exists(destination):
# 			source = os.path.join(INPUT_DIR + 'train', file_name)
# 			shutil.copyfile(source, destination)

print('step 4')
# Data augmentation
training_data_generator = ImageDataGenerator(rescale=1./255,
                                             horizontal_flip=True,
                                             vertical_flip=True,
                                             rotation_range=90,
                                             zoom_range=0.2, 
                                             width_shift_range=0.1,
                                             height_shift_range=0.1,
                                             shear_range=0.05,
                                             channel_shift_range=0.1)


print('step 5')
# Data generation
training_generator = training_data_generator.flow_from_directory(training_path,
                                                                 target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                                                 batch_size=BATCH_SIZE,
                                                                 class_mode='binary')
validation_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(validation_path,
                                                                              target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                                                              batch_size=BATCH_SIZE,
                                                                              class_mode='binary')


print('step 6')
# Model (LB 0.9558)
input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
inputs = Input(input_shape)

xception = Xception(include_top=False, input_shape=input_shape)  
nas_net = NASNetMobile(include_top=False, input_shape=input_shape)

outputs = Concatenate(axis=-1)([GlobalAveragePooling2D()(xception(inputs)),
                                GlobalAveragePooling2D()(nas_net(inputs))])
outputs = Dropout(0.5)(outputs)
outputs = Dense(1, activation='sigmoid')(outputs)

model = Model(inputs, outputs)
model.compile(optimizer=Adam(lr=0.0001, decay=0.00001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()


print('step 7')
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
SVG(model_to_dot(model).create(prog='dot', format='svg'))

from keras.utils.vis_utils import plot_model
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True, expand_nested=True)
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


print('step 8')
#  Training
history = model.fit_generator(training_generator,
                              steps_per_epoch=len(training_generator), 
                              validation_data=validation_generator,
                              validation_steps=len(validation_generator),
                              epochs=EPOCHS,
                              verbose=VERBOSITY,
                              callbacks=[PlotLossesKeras(),
                                         ModelCheckpoint(MODEL_FILE,
                                                         monitor='val_acc',
                                                         verbose=VERBOSITY,
                                                         save_best_only=True,
                                                         mode='max'),
                                         CSVLogger(TRAINING_LOGS_FILE,
                                                   append=False,
                                                   separator=';')])
model.load_weights(MODEL_FILE)


print('step 9')
# Training plots
epochs = [i for i in range(1, len(history.history['loss'])+1)]

plt.plot(epochs, history.history['loss'], color='blue', label="training_loss")
plt.plot(epochs, history.history['val_loss'], color='red', label="validation_loss")
plt.legend(loc='best')
plt.title('training')
plt.xlabel('epoch')
plt.savefig(TRAINING_PLOT_FILE, bbox_inches='tight')
plt.show()

plt.plot(epochs, history.history['acc'], color='blue', label="training_accuracy")
plt.plot(epochs, history.history['val_acc'], color='red',label="validation_accuracy")
plt.legend(loc='best')
plt.title('validation')
plt.xlabel('epoch')
plt.savefig(VALIDATION_PLOT_FILE, bbox_inches='tight')
plt.show()


print('step 10')
# ROC validation plot
roc_validation_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(validation_path,
                                                                                  target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                                                                  batch_size=BATCH_SIZE,
                                                                                  class_mode='binary',
                                                                                  shuffle=False)
predictions = model.predict_generator(roc_validation_generator, steps=len(roc_validation_generator), verbose=VERBOSITY)
false_positive_rate, true_positive_rate, threshold = roc_curve(roc_validation_generator.classes, predictions)
area_under_curve = auc(false_positive_rate, true_positive_rate)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(false_positive_rate, true_positive_rate, label='AUC = {:.3f}'.format(area_under_curve))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.savefig(ROC_PLOT_FILE, bbox_inches='tight')
plt.show()

print('step 11')
# Kaggle testing
testing_files = glob(os.path.join(INPUT_DIR+'test/','*.tif'))
submission = pd.DataFrame()
for index in range(0, len(testing_files), TESTING_BATCH_SIZE):
	data_frame = pd.DataFrame({'path': testing_files[index:index+TESTING_BATCH_SIZE]})
	data_frame['id'] = data_frame.path.map(lambda x: x.split('/')[3].split(".")[0])
	data_frame['image'] = data_frame['path'].map(imread)
	images = np.stack(data_frame.image, axis=0)
	predicted_labels = [model.predict(np.expand_dims(image/255.0, axis=0))[0][0] for image in images]
	predictions = np.array(predicted_labels)
	data_frame['label'] = predictions
	submission = pd.concat([submission, data_frame[["id", "label"]]])
submission.to_csv(KAGGLE_SUBMISSION_FILE, index=False, header=True)