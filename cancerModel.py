# all required imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
import seaborn as sns
import os
import glob
import plotly.graph_objects as go
import cv2
from PIL import Image
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from PIL import ImageFile
from keras.layers import Dense, Flatten , Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True
# obtaining data
root_folder = '../input/224-224-cervical-cancer-screening/kaggle'
trainingFolder = os.path.join(root_folder,'train', 'train')

type1Folder = os.path.join(trainingFolder, 'Type_1')
type2Folder = os.path.join(trainingFolder, 'Type_2')
type3Folder = os.path.join(trainingFolder, 'Type_3')

trainingFilesType1 = glob.glob(type1Folder+'/*.jpg')
trainingFilesType2 = glob.glob(type2Folder+'/*.jpg')
trainingFilesType3 = glob.glob(type3Folder+'/*.jpg')

type1Extra  =  glob.glob(os.path.join(root_folder, "additional_Type_1_v2", "Type_1")+'/*.jpg')
type2Extra  =  glob.glob(os.path.join(root_folder, "additional_Type_2_v2", "Type_2")+'/*.jpg')
type3Extra  =  glob.glob(os.path.join(root_folder, "additional_Type_3_v2", "Type_3")+'/*.jpg')

type1_files = trainingFilesType1 + type1Extra
type2_files = trainingFilesType2 + type2Extra
type3_files = trainingFilesType3 + type3Extra
# create dataframe of file and labels
combinedFiles = {'filepath': type1_files + type2_files + type3_files,
          'label': ['Type_1']* len(type1_files) + ['Type_2']* len(type2_files) + ['Type_3']* len(type3_files)}

filesDataframe = pd.DataFrame(combinedFiles).sample(frac=1, random_state= 1).reset_index(drop=True)
# typecount with bar
plt.figure(figsize = (15, 6))
sns.barplot(x= type_count['Num_Values'], y= type_count.index.to_list())
plt.title('Cervical Cancer Type Distribution')
plt.grid(True)
plt.show()
# label distribution pie
pie_plot = go.Pie(labels= type_count.index.to_list(), values= type_count.values.flatten(),
                 hole= 0.2, text= type_count.index.to_list(), textposition='auto')
fig = go.Figure([pie_plot])
fig.update_layout(title_text='Pie Plot of Type Distribution')
fig.show()
# sample images of types of ...
for label in ('Type_1', 'Type_2', 'Type_3'):
    filepaths = filesDataframe[filesDataframe['label']==label]['filepath'].values[:5]
    fig = plt.figure(figsize= (15, 6))
    for i, path in enumerate(filepaths):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (224, 224))
        fig.add_subplot(1, 5, i+1)
        plt.imshow(img)
        plt.subplots_adjust(hspace=0.5)
        plt.axis(False)
        plt.title(label)
#split the data into train  and validation set
train_df, eval_df = train_test_split(filesDataframe, test_size= 0.2, stratify= filesDataframe['label'], random_state= 1)
val_df, test_df = train_test_split(eval_df, test_size= 0.5, stratify= eval_df['label'], random_state= 1)
print(len(train_df), len(val_df), len(test_df))
# loads images from dataframe
def imageDataFrame(dataframe):
    features = []
    filepaths = dataframe['filepath'].values
    labels = dataframe['label'].values
    
    for path in filepaths:
        img = cv2.imread(path)
        resized_img = cv2.resize(img, (180, 180))
        features.append(np.array(resized_img))
    return np.array(features), np.array(labels)
train_features, train_labels = imageDataFrame(train_df)
val_features, val_labels = imageDataFrame(val_df)
test_features, test_labels = imageDataFrame(test_df)
#image shape
InputShape = train_features[0].shape
# normalize the features
X_train = train_features/250
X_val  = val_features/250
X_test  = test_features/250
# encoding the labels
le = LabelEncoder().fit(['Type_1', 'Type_2', 'Type_3'])
y_train = le.transform(train_labels)
y_val = le.transform(val_labels)
y_test = le.transform(test_labels)
# initialize image data generator for training and evaluation sets
train_datagen = ImageDataGenerator(
                                rotation_range = 40,
                                zoom_range = 0.2,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                shear_range=0.2,
                                horizontal_flip=True,
                                vertical_flip = True)

eval_datagen = ImageDataGenerator()
# data augmentation
BATCH_SIZE= 90
train_gen = train_datagen.flow(X_train, y_train, batch_size= BATCH_SIZE)
val_gen = eval_datagen.flow(X_val, y_val, batch_size= BATCH_SIZE)
test_gen = eval_datagen.flow(X_test, y_test, batch_size= BATCH_SIZE)
# initialize pretrained vgg model base
conv_base = VGG16(weights= 'imagenet', include_top= False, input_shape= (180, 180, 3))
# freeze few layers of pretrained model
for layer in conv_base.layers[:-5]:
    layer.trainable= False
    # build model 
model = Sequential([conv_base, 
                    Flatten(),
                   Dropout(0.5),
                   Dense(3, activation='softmax')])
# compile model
model.compile(optimizer= Adam(0.001), loss= 'sparse_categorical_crossentropy', metrics= ['accuracy'])
TRAIN_STEPS = len(train_df)//BATCH_SIZE
VAL_STEPS = len(val_df)//BATCH_SIZE
# initialize callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience = 20, verbose=1, mode='min', restore_best_weights= True)
reduceLR = ReduceLROnPlateau(monitor='val_loss', patience=10, verbose= 1, mode='min', factor=  0.2, min_lr = 1e-5)
checkpoint = ModelCheckpoint('cervicalModel.weights.hdf5', monitor='val_loss', verbose=1,save_best_only=True, mode= 'min')
# train model
history = model.fit(train_gen, steps_per_epoch= TRAIN_STEPS, validation_data=val_gen, validation_steps=VAL_STEPS, epochs= 1,
                   callbacks= [reduceLR, early_stopping, checkpoint])
# loading weights into model
model.load_weights('cervicalModel.weights.hdf5')
# save model
model.save('hub_cancer_screen_model.h5')
# evaluate model on test set
model.evaluate(test_gen)
# get test data directory
test_dir = os.path.join(root_folder,'test_stg2')
# load test features and labels
test_filenames = []
test_features = []
for filename in os.listdir(test_dir):
    test_filenames.append(filename)
    filepath = os.path.join(test_dir, filename)
    img = cv2.imread(filepath)
    resized_img = cv2.resize(img, (180, 180))
    test_features.append(np.array(resized_img)) 
    # normalize test features
test_X = np.array(test_features)
test_X = test_X/255
# get test predictions
test_predict = model.predict(test_X)
test_predict[0]
# create dataframe of test predictions
createDir = pd.DataFrame(test_predict, columns= ['Type_1', 'Type_2', 'Type_3' ])
createDir['image_name'] = test_filenames
createDir = createDir[['image_name', 'Type_1', 'Type_2', 'Type_3']]
createDir = createDir.sort_values(['image_name']).reset_index(drop=True)
createDir.head()
# creating csv file of test predictions
createDir.to_csv('submission_type1.csv', index=False)