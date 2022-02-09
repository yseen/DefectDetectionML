'''
Transfer learning for classification of springs and gears.
Adapted from https://www.analyticsvidhya.com/blog/2020/08/top-4-pre-trained-models-for-image-classification-with-python-code/
'''
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop

base_dir = './ClassificationDataSet' 
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# Directory with our training cat pictures
train_gears_dir = os.path.join(train_dir, 'gears')

# Directory with our training dog pictures
train_springs_dir = os.path.join(train_dir, 'springs')

# Directory with our validation cat pictures
validation_gears_dir = os.path.join(validation_dir, 'gears')

# Directory with our validation dog pictures
validation_springs_dir = os.path.join(validation_dir, 'springs')

# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator()

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator()

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(train_dir, batch_size = 20, class_mode = 'binary', target_size = (224, 224))

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = test_datagen.flow_from_directory( validation_dir,  batch_size = 20, class_mode = 'binary', target_size = (224, 224))

#load base model, without top

base_model = VGG16(input_shape = (224, 224, 3), # Shape of our images
include_top = False, # Leave out the last fully connected layer
weights = 'imagenet')

#set base layers to untrainable
for layer in base_model.layers:
    layer.trainable = False
    
#compile, add final layer
# Flatten the output layer to 1 dimension
x = layers.Flatten()(base_model.output)

# Add a fully connected layer with 512 hidden units and ReLU activation
x = layers.Dense(512, activation='relu')(x)

# Add a dropout rate of 0.5
x = layers.Dropout(0.5)(x)

# Add a final sigmoid layer with 1 node for classification output
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(base_model.input, x)

model.compile(optimizer = RMSprop(lr=0.0001), loss = 'binary_crossentropy',metrics = ['acc'])

#fit model
vgghist = model.fit(train_generator, validation_data = validation_generator, steps_per_epoch = 10, epochs = 1)

#save
model.save('C:\\GitHub\\DefectDetectionML\\ClassificationDataSet\\classE1.pt')
np.save('C:\\GitHub\\DefectDetectionML\\ClassificationDataSet\\historyE1.npy',vgghist.history)