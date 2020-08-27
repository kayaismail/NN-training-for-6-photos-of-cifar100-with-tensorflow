import keras
import os, shutil
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import models
from keras import optimizers
from keras.layers.core import Dropout
from keras.utils import to_categorical
import matplotlib.pyplot as plt

keras.__version__

#indirdigimiz dosya DATASET
original_dataset_train_cattle_dir = 'TRAIN/cattle'
original_dataset_test_cattle_dir = 'TEST/cattle'
original_dataset_train_elephant_dir = 'TRAIN/elephant'
original_dataset_test_elephant_dir = 'TEST/elephant'
original_dataset_train_fox_dir = 'TRAIN/fox'
original_dataset_test_fox_dir = 'TEST/fox'
original_dataset_train_leopard_dir = 'TRAIN/leopard'
original_dataset_test_leopard_dir = 'TEST/leopard'
original_dataset_train_shark_dir = 'TRAIN/shark'
original_dataset_test_shark_dir = 'TEST/shark'
original_dataset_train_table_dir = 'TRAIN/table'
original_dataset_test_table_dir = 'TEST/table'
#kendi dosyamiz  store our

base_dir = '/Users/ismailkaya/Downloads/Compressed/Benimsinifim'
os.mkdir(base_dir)
# validation and test splits,bolumlerimiz
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

# Directory with our training cattle pictures
train_cattle_dir = os.path.join(train_dir, 'cattle')
os.mkdir(train_cattle_dir)
# Directory with our training elephant pictures
train_elephant_dir = os.path.join(train_dir, 'elephant')
os.mkdir(train_elephant_dir)
# Directory with our training fox pictures
train_fox_dir = os.path.join(train_dir, 'fox')
os.mkdir(train_fox_dir)
# Directory with our training leopard pictures
train_leopard_dir = os.path.join(train_dir, 'leopard')
os.mkdir(train_leopard_dir)
# Directory with our training shark  pictures
train_shark_dir = os.path.join(train_dir, 'shark')
os.mkdir(train_shark_dir)
# Directory with our training table pictures
train_table_dir = os.path.join(train_dir, 'table')
os.mkdir(train_table_dir)
#--------------------------------------------------------------
# Directory with our validation cattle pictures
validation_cattle_dir = os.path.join(validation_dir, 'cattle')
os.mkdir(validation_cattle_dir)

# Directory with our validation elephant pictures
validation_elephant_dir = os.path.join(validation_dir, 'elephant')
os.mkdir(validation_elephant_dir)
# Directory with our validation fox pictures
validation_fox_dir = os.path.join(validation_dir, 'fox')
os.mkdir(validation_fox_dir)

# Directory with our validation leopard pictures
validation_leopard_dir = os.path.join(validation_dir, 'leopard')
os.mkdir(validation_leopard_dir)
# Directory with our validation shark pictures
validation_shark_dir = os.path.join(validation_dir, 'shark')
os.mkdir(validation_shark_dir)

# Directory with our validation dog pictures
validation_table_dir = os.path.join(validation_dir, 'table')
os.mkdir(validation_table_dir)
#--------------------------------------------------------------
# Directory with our test cat pictures
test_cattle_dir = os.path.join(test_dir, 'cattle')
os.mkdir(test_cattle_dir)

# Directory with our test dog pictures
test_elephant_dir = os.path.join(test_dir, 'elephant')
os.mkdir(test_elephant_dir)
# Directory with our test fox pictures
test_fox_dir = os.path.join(test_dir, 'fox')
os.mkdir(test_fox_dir)

# Directory with our test leopard pictures
test_leopard_dir = os.path.join(test_dir, 'leopard')
os.mkdir(test_leopard_dir)
# Directory with our test shark pictures
test_shark_dir = os.path.join(test_dir, 'shark')
os.mkdir(test_shark_dir)

# Directory with our test table pictures
test_table_dir = os.path.join(test_dir, 'table')
os.mkdir(test_table_dir)
#---------------------------------------------------------------------------------------
# Copy first 500 cattle images to train_cattle_dir
fnames = ['c ({}).png'.format(i) for i in range(1,501)]
for fname in fnames:
    src = os.path.join(original_dataset_train_cattle_dir, fname)
    dst = os.path.join(train_cattle_dir, fname)
    shutil.copyfile(src, dst)

# Copy next 100 cattle images to validation_cattle_dir
fnames = ['c ({}).png'.format(i) for i in range(1,101)]
for fname in fnames:
    src = os.path.join(original_dataset_test_cattle_dir, fname)
    dst = os.path.join(validation_cattle_dir, fname)
    shutil.copyfile(src, dst)
    
# Copy next 100 cattle images to test_cattle_dir
fnames = ['c ({}).png'.format(i) for i in range(1,101)]
for fname in fnames:
    src = os.path.join(original_dataset_test_cattle_dir, fname)
    dst = os.path.join(test_cattle_dir, fname)
    shutil.copyfile(src, dst)
#----------------------********************
# Copy first 500 elephant images to train_elephant_dir
fnames = ['e ({}).png'.format(i) for i in range(1,501)]
for fname in fnames:
    src = os.path.join(original_dataset_train_elephant_dir, fname)
    dst = os.path.join(train_elephant_dir, fname)
    shutil.copyfile(src, dst)

# Copy next 100 elephant images to validation_elephant_dir
fnames = ['e ({}).png'.format(i) for i in range(1,101)]
for fname in fnames:
    src = os.path.join(original_dataset_test_elephant_dir, fname)
    dst = os.path.join(validation_elephant_dir, fname)
    shutil.copyfile(src, dst)
    
# Copy next 100 elephant images to test_elephant_dir
fnames = ['e ({}).png'.format(i) for i in range(1,101)]
for fname in fnames:
    src = os.path.join(original_dataset_test_elephant_dir, fname)
    dst = os.path.join(test_elephant_dir, fname)
    shutil.copyfile(src, dst)
#----------------------*******************
# Copy first 500 fox images to train_fox_dir
fnames = ['f ({}).png'.format(i) for i in range(1,501)]
for fname in fnames:
    src = os.path.join(original_dataset_train_fox_dir, fname)
    dst = os.path.join(train_fox_dir, fname)
    shutil.copyfile(src, dst)

# Copy next 100 fox images to validation_fox_dir
fnames = ['f ({}).png'.format(i) for i in range(1,101)]
for fname in fnames:
    src = os.path.join(original_dataset_test_fox_dir, fname)
    dst = os.path.join(validation_fox_dir, fname)
    shutil.copyfile(src, dst)
    
# Copy next 100 fox images to test_fox_dir
fnames = ['f ({}).png'.format(i) for i in range(1,101)]
for fname in fnames:
    src = os.path.join(original_dataset_test_fox_dir, fname)
    dst = os.path.join(test_fox_dir, fname)
    shutil.copyfile(src, dst)
#----------------------*******************
# Copy first 500 leopard images to train_leopard_dir
fnames = ['l ({}).png'.format(i) for i in range(1,501)]
for fname in fnames:
    src = os.path.join(original_dataset_train_leopard_dir, fname)
    dst = os.path.join(train_leopard_dir, fname)
    shutil.copyfile(src, dst)

# Copy next 100 leopard images to validation_leopard_dir
fnames = ['l ({}).png'.format(i) for i in range(1,101)]
for fname in fnames:
    src = os.path.join(original_dataset_test_leopard_dir, fname)
    dst = os.path.join(validation_leopard_dir, fname)
    shutil.copyfile(src, dst)
    
# Copy next 100 leopard images to test_leopard_dir
fnames = ['l ({}).png'.format(i) for i in range(1,101)]
for fname in fnames:
    src = os.path.join(original_dataset_test_leopard_dir, fname)
    dst = os.path.join(test_leopard_dir, fname)
    shutil.copyfile(src, dst)
#----------------------*******************
# Copy first 500 shark images to train_shark_dir
fnames = ['s ({}).png'.format(i) for i in range(1,501)]
for fname in fnames:
    src = os.path.join(original_dataset_train_shark_dir, fname)
    dst = os.path.join(train_shark_dir, fname)
    shutil.copyfile(src, dst)

# Copy next 100 shark images to validation_shark_dir
fnames = ['s ({}).png'.format(i) for i in range(1,101)]
for fname in fnames:
    src = os.path.join(original_dataset_test_shark_dir, fname)
    dst = os.path.join(validation_shark_dir, fname)
    shutil.copyfile(src, dst)
    
# Copy next 100 shark images to test_shark_dir
fnames = ['s ({}).png'.format(i) for i in range(1,101)]
for fname in fnames:
    src = os.path.join(original_dataset_test_shark_dir, fname)
    dst = os.path.join(test_shark_dir, fname)
    shutil.copyfile(src, dst)
#----------------------*******************
# Copy first 500 table images to train_table_dir
fnames = ['t ({}).png'.format(i) for i in range(1,501)]
for fname in fnames:
    src = os.path.join(original_dataset_train_table_dir, fname)
    dst = os.path.join(train_table_dir, fname)
    shutil.copyfile(src, dst)

# Copy next 100 table images to validation_table_dir
fnames = ['t ({}).png'.format(i) for i in range(1,101)]
for fname in fnames:
    src = os.path.join(original_dataset_test_table_dir, fname)
    dst = os.path.join(validation_table_dir, fname)
    shutil.copyfile(src, dst)
    
# Copy next 100 table images to test_table_dir
fnames = ['t ({}).png'.format(i) for i in range(1,101)]
for fname in fnames:
    src = os.path.join(original_dataset_test_table_dir, fname)
    dst = os.path.join(test_table_dir, fname)
    shutil.copyfile(src, dst)
    
#*********************************************************************
 
model = models.Sequential()
model.add(layers.Conv2D(16,(3, 3), activation='relu',
                        padding='same',
                        input_shape=(32, 32, 3)))
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        padding='same')) 
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64, (3, 3), activation='relu',
                        padding='same'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(128, (3, 3), activation='relu',
                        padding='same'))
model.add(layers.MaxPooling2D())
model.add(layers.Flatten())   # dizi haline getirmek icin 
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(6, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(), 
              metrics=['acc'])

# All images will be rescaled by 1./255 

train_datagen = ImageDataGenerator(rescale=1./255,
                                   horizontal_flip=True,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(32, 32),
        batch_size=200,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(32, 32),
        batch_size=10,
        class_mode='categorical') 

history = model.fit_generator(
      train_generator,
      steps_per_epoch=25,
     epochs=30,
     validation_data=validation_generator,
      validation_steps=30)

#grafiklerimizi cizdirelim 
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

plt.show()