# 人工智慧課程期末報告-使用類似 U-Net 的架構進行影像分割

## 首先先從網站下載範例的訓練資料
```!!wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
!!wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
!
!curl -O https://thor.robots.ox.ac.uk/datasets/pets/images.tar.gz
!curl -O https://thor.robots.ox.ac.uk/datasets/pets/annotations.tar.gz
!
!tar -xf images.tar.gz
!tar -xf annotations.tar.gz
```
!!wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
!!wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
!
!curl -O https://thor.robots.ox.ac.uk/datasets/pets/images.tar.gz
!curl -O https://thor.robots.ox.ac.uk/datasets/pets/annotations.tar.gz
!
!tar -xf images.tar.gz
!tar -xf annotations.tar.gz

##取得訓練用的圖片和目標圖片遮罩的檔案路徑
```import os

input_dir = "images/"
target_dir = "annotations/trimaps/"
img_size = (160, 160)
num_classes = 3
batch_size = 32

input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".jpg")
    ]
)
target_img_paths = sorted(
    [
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.endswith(".png") and not fname.startswith(".")
    ]
)

print("Number of samples:", len(input_img_paths))

for input_path, target_path in zip(input_img_paths[:10], target_img_paths[:10]):
    print(input_path, "|", target_path)
```
Number of samples: 7390
images/Abyssinian_1.jpg | annotations/trimaps/Abyssinian_1.png
images/Abyssinian_10.jpg | annotations/trimaps/Abyssinian_10.png
images/Abyssinian_100.jpg | annotations/trimaps/Abyssinian_100.png
images/Abyssinian_101.jpg | annotations/trimaps/Abyssinian_101.png
images/Abyssinian_102.jpg | annotations/trimaps/Abyssinian_102.png
images/Abyssinian_103.jpg | annotations/trimaps/Abyssinian_103.png
images/Abyssinian_104.jpg | annotations/trimaps/Abyssinian_104.png
images/Abyssinian_105.jpg | annotations/trimaps/Abyssinian_105.png
images/Abyssinian_106.jpg | annotations/trimaps/Abyssinian_106.png
images/Abyssinian_107.jpg | annotations/trimaps/Abyssinian_107.png

##顯示其中一張圖片和其對應的遮罩
```from IPython.display import Image, display
from keras.utils import load_img
from PIL import ImageOps

# Display input image #7
display(Image(filename=input_img_paths[9]))

# Display auto-contrast version of corresponding target (per-pixel categories)
img = ImageOps.autocontrast(load_img(target_img_paths[9]))
display(img)
```
![image](https://github.com/10824209minggui/Finalreport/assets/149359020/62b5d1ac-2f84-4498-8830-a84226e7fcb1)

![image](https://github.com/10824209minggui/Finalreport/assets/149359020/7696ea9e-5754-43eb-a37a-a429bca521a0)

##定義類別以一次傳回多個資料
```import keras
import numpy as np
from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow import io as tf_io


def get_dataset(
    batch_size,
    img_size,
    input_img_paths,
    target_img_paths,
    max_dataset_len=None,
):
    """Returns a TF Dataset."""

    def load_img_masks(input_img_path, target_img_path):
        input_img = tf_io.read_file(input_img_path)
        input_img = tf_io.decode_png(input_img, channels=3)
        input_img = tf_image.resize(input_img, img_size)
        input_img = tf_image.convert_image_dtype(input_img, "float32")

        target_img = tf_io.read_file(target_img_path)
        target_img = tf_io.decode_png(target_img, channels=1)
        target_img = tf_image.resize(target_img, img_size, method="nearest")
        target_img = tf_image.convert_image_dtype(target_img, "uint8")

        # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
        target_img -= 1
        return input_img, target_img

    # For faster debugging, limit the size of data
    if max_dataset_len:
        input_img_paths = input_img_paths[:max_dataset_len]
        target_img_paths = target_img_paths[:max_dataset_len]
    dataset = tf_data.Dataset.from_tensor_slices((input_img_paths, target_img_paths))
    dataset = dataset.map(load_img_masks, num_parallel_calls=tf_data.AUTOTUNE)
    return dataset.batch(batch_size)
```

##建立 U-Net Xception-style 模型如下
```from keras import layers


def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model


# Build model
model = get_model(img_size, num_classes)
model.summary()
```
Model: "model"
__________________________________________________________________________________________________
 Layer (type)                Output Shape                 Param #   Connected to                  
==================================================================================================
 input_1 (InputLayer)        [(None, 160, 160, 3)]        0         []                            
                                                                                                  
 conv2d (Conv2D)             (None, 80, 80, 32)           896       ['input_1[0][0]']             
                                                                                                  
 batch_normalization (Batch  (None, 80, 80, 32)           128       ['conv2d[0][0]']              
 Normalization)                                                                                   
                                                                                                  
 activation (Activation)     (None, 80, 80, 32)           0         ['batch_normalization[0][0]'] 
                                                                                                  
 activation_1 (Activation)   (None, 80, 80, 32)           0         ['activation[0][0]']          
                                                                                                  
 separable_conv2d (Separabl  (None, 80, 80, 64)           2400      ['activation_1[0][0]']        
 eConv2D)                                                                                         
                                                                                                  
 batch_normalization_1 (Bat  (None, 80, 80, 64)           256       ['separable_conv2d[0][0]']    
 chNormalization)                                                                                 
                                                                                                  
 activation_2 (Activation)   (None, 80, 80, 64)           0         ['batch_normalization_1[0][0]'
                                                                    ]                             
                                                                                                  
 separable_conv2d_1 (Separa  (None, 80, 80, 64)           4736      ['activation_2[0][0]']        
 bleConv2D)                                                                                       
                                                                                                  
 batch_normalization_2 (Bat  (None, 80, 80, 64)           256       ['separable_conv2d_1[0][0]']  
 chNormalization)                                                                                 
                                                                                                  
 max_pooling2d (MaxPooling2  (None, 40, 40, 64)           0         ['batch_normalization_2[0][0]'
 D)                                                                 ]                             
                                                                                                  
 conv2d_1 (Conv2D)           (None, 40, 40, 64)           2112      ['activation[0][0]']          
                                                                                                  
 add (Add)                   (None, 40, 40, 64)           0         ['max_pooling2d[0][0]',       
                                                                     'conv2d_1[0][0]']            
                                                                                                  
 activation_3 (Activation)   (None, 40, 40, 64)           0         ['add[0][0]']                 
                                                                                                  
 separable_conv2d_2 (Separa  (None, 40, 40, 128)          8896      ['activation_3[0][0]']        
 bleConv2D)                                                                                       
                                                                                                  
 batch_normalization_3 (Bat  (None, 40, 40, 128)          512       ['separable_conv2d_2[0][0]']  
 chNormalization)                                                                                 
                                                                                                  
 activation_4 (Activation)   (None, 40, 40, 128)          0         ['batch_normalization_3[0][0]'
                                                                    ]                             
                                                                                                  
 separable_conv2d_3 (Separa  (None, 40, 40, 128)          17664     ['activation_4[0][0]']        
 bleConv2D)                                                                                       
                                                                                                  
 batch_normalization_4 (Bat  (None, 40, 40, 128)          512       ['separable_conv2d_3[0][0]']  
 chNormalization)                                                                                 
                                                                                                  
 max_pooling2d_1 (MaxPoolin  (None, 20, 20, 128)          0         ['batch_normalization_4[0][0]'
 g2D)                                                               ]                             
                                                                                                  
 conv2d_2 (Conv2D)           (None, 20, 20, 128)          8320      ['add[0][0]']                 
                                                                                                  
 add_1 (Add)                 (None, 20, 20, 128)          0         ['max_pooling2d_1[0][0]',     
                                                                     'conv2d_2[0][0]']            
                                                                                                  
 activation_5 (Activation)   (None, 20, 20, 128)          0         ['add_1[0][0]']               
                                                                                                  
 separable_conv2d_4 (Separa  (None, 20, 20, 256)          34176     ['activation_5[0][0]']        
 bleConv2D)                                                                                       
                                                                                                  
 batch_normalization_5 (Bat  (None, 20, 20, 256)          1024      ['separable_conv2d_4[0][0]']  
 chNormalization)                                                                                 
                                                                                                  
 activation_6 (Activation)   (None, 20, 20, 256)          0         ['batch_normalization_5[0][0]'
                                                                    ]                             
                                                                                                  
 separable_conv2d_5 (Separa  (None, 20, 20, 256)          68096     ['activation_6[0][0]']        
 bleConv2D)                                                                                       
                                                                                                  
 batch_normalization_6 (Bat  (None, 20, 20, 256)          1024      ['separable_conv2d_5[0][0]']  
 chNormalization)                                                                                 
                                                                                                  
 max_pooling2d_2 (MaxPoolin  (None, 10, 10, 256)          0         ['batch_normalization_6[0][0]'
 g2D)                                                               ]                             
                                                                                                  
 conv2d_3 (Conv2D)           (None, 10, 10, 256)          33024     ['add_1[0][0]']               
                                                                                                  
 add_2 (Add)                 (None, 10, 10, 256)          0         ['max_pooling2d_2[0][0]',     
                                                                     'conv2d_3[0][0]']            
                                                                                                  
 activation_7 (Activation)   (None, 10, 10, 256)          0         ['add_2[0][0]']               
                                                                                                  
 conv2d_transpose (Conv2DTr  (None, 10, 10, 256)          590080    ['activation_7[0][0]']        
 anspose)                                                                                         
                                                                                                  
 batch_normalization_7 (Bat  (None, 10, 10, 256)          1024      ['conv2d_transpose[0][0]']    
 chNormalization)                                                                                 
                                                                                                  
 activation_8 (Activation)   (None, 10, 10, 256)          0         ['batch_normalization_7[0][0]'
                                                                    ]                             
                                                                                                  
 conv2d_transpose_1 (Conv2D  (None, 10, 10, 256)          590080    ['activation_8[0][0]']        
 Transpose)                                                                                       
                                                                                                  
 batch_normalization_8 (Bat  (None, 10, 10, 256)          1024      ['conv2d_transpose_1[0][0]']  
 chNormalization)                                                                                 
                                                                                                  
 up_sampling2d_1 (UpSamplin  (None, 20, 20, 256)          0         ['add_2[0][0]']               
 g2D)                                                                                             
                                                                                                  
 up_sampling2d (UpSampling2  (None, 20, 20, 256)          0         ['batch_normalization_8[0][0]'
 D)                                                                 ]                             
                                                                                                  
 conv2d_4 (Conv2D)           (None, 20, 20, 256)          65792     ['up_sampling2d_1[0][0]']     
                                                                                                  
 add_3 (Add)                 (None, 20, 20, 256)          0         ['up_sampling2d[0][0]',       
                                                                     'conv2d_4[0][0]']            
                                                                                                  
 activation_9 (Activation)   (None, 20, 20, 256)          0         ['add_3[0][0]']               
                                                                                                  
 conv2d_transpose_2 (Conv2D  (None, 20, 20, 128)          295040    ['activation_9[0][0]']        
 Transpose)                                                                                       
                                                                                                  
 batch_normalization_9 (Bat  (None, 20, 20, 128)          512       ['conv2d_transpose_2[0][0]']  
 chNormalization)                                                                                 
                                                                                                  
 activation_10 (Activation)  (None, 20, 20, 128)          0         ['batch_normalization_9[0][0]'
                                                                    ]                             
                                                                                                  
 conv2d_transpose_3 (Conv2D  (None, 20, 20, 128)          147584    ['activation_10[0][0]']       
 Transpose)                                                                                       
                                                                                                  
 batch_normalization_10 (Ba  (None, 20, 20, 128)          512       ['conv2d_transpose_3[0][0]']  
 tchNormalization)                                                                                
                                                                                                  
 up_sampling2d_3 (UpSamplin  (None, 40, 40, 256)          0         ['add_3[0][0]']               
 g2D)                                                                                             
                                                                                                  
 up_sampling2d_2 (UpSamplin  (None, 40, 40, 128)          0         ['batch_normalization_10[0][0]
 g2D)                                                               ']                            
                                                                                                  
 conv2d_5 (Conv2D)           (None, 40, 40, 128)          32896     ['up_sampling2d_3[0][0]']     
                                                                                                  
 add_4 (Add)                 (None, 40, 40, 128)          0         ['up_sampling2d_2[0][0]',     
                                                                     'conv2d_5[0][0]']            
                                                                                                  
 activation_11 (Activation)  (None, 40, 40, 128)          0         ['add_4[0][0]']               
                                                                                                  
 conv2d_transpose_4 (Conv2D  (None, 40, 40, 64)           73792     ['activation_11[0][0]']       
 Transpose)                                                                                       
                                                                                                  
 batch_normalization_11 (Ba  (None, 40, 40, 64)           256       ['conv2d_transpose_4[0][0]']  
 tchNormalization)                                                                                
                                                                                                  
 activation_12 (Activation)  (None, 40, 40, 64)           0         ['batch_normalization_11[0][0]
                                                                    ']                            
                                                                                                  
 conv2d_transpose_5 (Conv2D  (None, 40, 40, 64)           36928     ['activation_12[0][0]']       
 Transpose)                                                                                       
                                                                                                  
 batch_normalization_12 (Ba  (None, 40, 40, 64)           256       ['conv2d_transpose_5[0][0]']  
 tchNormalization)                                                                                
                                                                                                  
 up_sampling2d_5 (UpSamplin  (None, 80, 80, 128)          0         ['add_4[0][0]']               
 g2D)                                                                                             
                                                                                                  
 up_sampling2d_4 (UpSamplin  (None, 80, 80, 64)           0         ['batch_normalization_12[0][0]
 g2D)                                                               ']                            
                                                                                                  
 conv2d_6 (Conv2D)           (None, 80, 80, 64)           8256      ['up_sampling2d_5[0][0]']     
                                                                                                  
 add_5 (Add)                 (None, 80, 80, 64)           0         ['up_sampling2d_4[0][0]',     
                                                                     'conv2d_6[0][0]']            
                                                                                                  
 activation_13 (Activation)  (None, 80, 80, 64)           0         ['add_5[0][0]']               
                                                                                                  
 conv2d_transpose_6 (Conv2D  (None, 80, 80, 32)           18464     ['activation_13[0][0]']       
 Transpose)                                                                                       
                                                                                                  
 batch_normalization_13 (Ba  (None, 80, 80, 32)           128       ['conv2d_transpose_6[0][0]']  
 tchNormalization)                                                                                
                                                                                                  
 activation_14 (Activation)  (None, 80, 80, 32)           0         ['batch_normalization_13[0][0]
                                                                    ']                            
                                                                                                  
 conv2d_transpose_7 (Conv2D  (None, 80, 80, 32)           9248      ['activation_14[0][0]']       
 Transpose)                                                                                       
                                                                                                  
 batch_normalization_14 (Ba  (None, 80, 80, 32)           128       ['conv2d_transpose_7[0][0]']  
 tchNormalization)                                                                                
                                                                                                  
 up_sampling2d_7 (UpSamplin  (None, 160, 160, 64)         0         ['add_5[0][0]']               
 g2D)                                                                                             
                                                                                                  
 up_sampling2d_6 (UpSamplin  (None, 160, 160, 32)         0         ['batch_normalization_14[0][0]
 g2D)                                                               ']                            
                                                                                                  
 conv2d_7 (Conv2D)           (None, 160, 160, 32)         2080      ['up_sampling2d_7[0][0]']     
                                                                                                  
 add_6 (Add)                 (None, 160, 160, 32)         0         ['up_sampling2d_6[0][0]',     
                                                                     'conv2d_7[0][0]']            
                                                                                                  
 conv2d_8 (Conv2D)           (None, 160, 160, 3)          867       ['add_6[0][0]']               
                                                                                                  
==================================================================================================
Total params: 2058979 (7.85 MB)
Trainable params: 2055203 (7.84 MB)
Non-trainable params: 3776 (14.75 KB)
__________________________________________________________________________________________________

##將資料切割為訓練及驗證資料
```import random

# Split our img paths into a training and a validation set
val_samples = 1000
random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_img_paths)
train_input_img_paths = input_img_paths[:-val_samples]
train_target_img_paths = target_img_paths[:-val_samples]
val_input_img_paths = input_img_paths[-val_samples:]
val_target_img_paths = target_img_paths[-val_samples:]

# Instantiate dataset for each split
# Limit input files in `max_dataset_len` for faster epoch training time.
# Remove the `max_dataset_len` arg when running with full dataset.
train_dataset = get_dataset(
    batch_size,
    img_size,
    train_input_img_paths,
    train_target_img_paths,
    max_dataset_len=1000,
)
valid_dataset = get_dataset(
    batch_size, img_size, val_input_img_paths, val_target_img_paths
)
```

##訓練模型
```# Configure the model for training.
# We use the "sparse" version of categorical_crossentropy
# because our target data is integers.
model.compile(
    optimizer=keras.optimizers.Adam(1e-4), loss="sparse_categorical_crossentropy"
)

callbacks = [
    keras.callbacks.ModelCheckpoint("oxford_segmentation.keras", save_best_only=True)
]

# Train the model, doing validation at the end of each epoch.
epochs = 50
model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=valid_dataset,
    callbacks=callbacks,
    verbose=2,
)
```
Epoch 1/50
32/32 - 350s - loss: 1.8899 - val_loss: 1.4152 - 350s/epoch - 11s/step
Epoch 2/50
32/32 - 330s - loss: 0.9626 - val_loss: 1.5932 - 330s/epoch - 10s/step
Epoch 3/50
32/32 - 323s - loss: 0.8017 - val_loss: 2.4171 - 323s/epoch - 10s/step
Epoch 4/50
32/32 - 309s - loss: 0.7204 - val_loss: 3.2333 - 309s/epoch - 10s/step
Epoch 5/50
32/32 - 311s - loss: 0.6659 - val_loss: 3.9848 - 311s/epoch - 10s/step
Epoch 6/50
32/32 - 307s - loss: 0.6238 - val_loss: 4.6861 - 307s/epoch - 10s/step
Epoch 7/50
32/32 - 318s - loss: 0.5882 - val_loss: 5.3560 - 318s/epoch - 10s/step
Epoch 8/50
32/32 - 309s - loss: 0.5566 - val_loss: 6.0104 - 309s/epoch - 10s/step
Epoch 9/50
32/32 - 319s - loss: 0.5272 - val_loss: 6.6448 - 319s/epoch - 10s/step
Epoch 10/50
32/32 - 320s - loss: 0.4992 - val_loss: 7.2713 - 320s/epoch - 10s/step
Epoch 11/50
32/32 - 325s - loss: 0.4719 - val_loss: 7.8212 - 325s/epoch - 10s/step
Epoch 12/50
32/32 - 319s - loss: 0.4455 - val_loss: 8.2856 - 319s/epoch - 10s/step
Epoch 13/50
32/32 - 319s - loss: 0.4198 - val_loss: 8.6095 - 319s/epoch - 10s/step
Epoch 14/50
32/32 - 321s - loss: 0.3951 - val_loss: 8.6664 - 321s/epoch - 10s/step
Epoch 15/50
32/32 - 320s - loss: 0.3723 - val_loss: 8.4205 - 320s/epoch - 10s/step
Epoch 16/50
32/32 - 321s - loss: 0.3520 - val_loss: 7.8449 - 321s/epoch - 10s/step
Epoch 17/50
32/32 - 307s - loss: 0.3369 - val_loss: 6.9301 - 307s/epoch - 10s/step
Epoch 18/50
32/32 - 323s - loss: 0.3408 - val_loss: 5.6162 - 323s/epoch - 10s/step
Epoch 19/50
32/32 - 320s - loss: 0.4261 - val_loss: 2.1721 - 320s/epoch - 10s/step
Epoch 20/50
32/32 - 319s - loss: 0.3990 - val_loss: 1.7679 - 319s/epoch - 10s/step
Epoch 21/50
32/32 - 313s - loss: 0.3479 - val_loss: 1.2122 - 313s/epoch - 10s/step
Epoch 22/50
32/32 - 319s - loss: 0.3339 - val_loss: 0.8666 - 319s/epoch - 10s/step
Epoch 23/50
32/32 - 319s - loss: 0.3353 - val_loss: 0.8441 - 319s/epoch - 10s/step
Epoch 24/50
32/32 - 304s - loss: 0.3448 - val_loss: 1.1430 - 304s/epoch - 9s/step
Epoch 25/50
32/32 - 319s - loss: 0.3585 - val_loss: 1.5595 - 319s/epoch - 10s/step
Epoch 26/50
32/32 - 316s - loss: 0.3464 - val_loss: 0.8759 - 316s/epoch - 10s/step
Epoch 27/50
32/32 - 320s - loss: 0.3151 - val_loss: 0.8772 - 320s/epoch - 10s/step
Epoch 28/50
32/32 - 320s - loss: 0.3063 - val_loss: 0.8516 - 320s/epoch - 10s/step
Epoch 29/50
32/32 - 321s - loss: 0.2998 - val_loss: 0.9038 - 321s/epoch - 10s/step
Epoch 30/50
32/32 - 321s - loss: 0.2993 - val_loss: 0.8417 - 321s/epoch - 10s/step
Epoch 31/50
32/32 - 317s - loss: 0.2795 - val_loss: 0.8906 - 317s/epoch - 10s/step
Epoch 32/50
32/32 - 318s - loss: 0.2553 - val_loss: 1.0448 - 318s/epoch - 10s/step
Epoch 33/50
32/32 - 302s - loss: 0.2504 - val_loss: 0.9539 - 302s/epoch - 9s/step
Epoch 34/50
32/32 - 315s - loss: 0.2559 - val_loss: 0.9056 - 315s/epoch - 10s/step
Epoch 35/50
32/32 - 316s - loss: 0.2642 - val_loss: 0.9460 - 316s/epoch - 10s/step
Epoch 36/50
32/32 - 298s - loss: 0.2579 - val_loss: 0.9095 - 298s/epoch - 9s/step
Epoch 37/50
32/32 - 315s - loss: 0.2442 - val_loss: 0.9262 - 315s/epoch - 10s/step
Epoch 38/50
32/32 - 315s - loss: 0.2307 - val_loss: 1.0812 - 315s/epoch - 10s/step
Epoch 39/50
32/32 - 305s - loss: 0.2223 - val_loss: 1.2457 - 305s/epoch - 10s/step
Epoch 40/50
32/32 - 314s - loss: 0.2073 - val_loss: 1.1898 - 314s/epoch - 10s/step
Epoch 41/50
32/32 - 300s - loss: 0.2047 - val_loss: 1.1159 - 300s/epoch - 9s/step
Epoch 42/50
32/32 - 315s - loss: 0.2056 - val_loss: 1.0962 - 315s/epoch - 10s/step
Epoch 43/50
32/32 - 319s - loss: 0.2081 - val_loss: 1.1228 - 319s/epoch - 10s/step
Epoch 44/50
32/32 - 301s - loss: 0.2104 - val_loss: 1.2577 - 301s/epoch - 9s/step
Epoch 45/50
32/32 - 315s - loss: 0.2197 - val_loss: 1.2157 - 315s/epoch - 10s/step
Epoch 46/50
32/32 - 314s - loss: 0.2228 - val_loss: 1.1518 - 314s/epoch - 10s/step
Epoch 47/50
32/32 - 306s - loss: 0.2247 - val_loss: 1.0218 - 306s/epoch - 10s/step
Epoch 48/50
32/32 - 315s - loss: 0.2202 - val_loss: 1.0113 - 315s/epoch - 10s/step
Epoch 49/50
32/32 - 315s - loss: 0.2126 - val_loss: 1.0429 - 315s/epoch - 10s/step
Epoch 50/50
32/32 - 317s - loss: 0.2089 - val_loss: 1.1143 - 317s/epoch - 10s/step
<keras.src.callbacks.History at 0x787dec270a90>

##進行預測，並顯示結果
```# Generate predictions for all images in the validation set

val_dataset = get_dataset(
    batch_size, img_size, val_input_img_paths, val_target_img_paths
)
val_preds = model.predict(val_dataset)


def display_mask(i):
    """Quick utility to display a model's prediction."""
    mask = np.argmax(val_preds[i], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    img = ImageOps.autocontrast(keras.utils.array_to_img(mask))
    display(img)


# Display results for validation image #10
i = 10

# Display input image
display(Image(filename=val_input_img_paths[i]))

# Display ground-truth target mask
img = ImageOps.autocontrast(load_img(val_target_img_paths[i]))
display(img)

# Display mask predicted by our model
display_mask(i)  # Note that the model only sees inputs at 150x150.

```
32/32 [==============================] - 70s 2s/step

##結果如下
![image](https://github.com/10824209minggui/Finalreport/assets/149359020/f7f86032-8397-4fa0-b282-057894aabfe7)

![image](https://github.com/10824209minggui/Finalreport/assets/149359020/56c72dcb-600a-4996-9be9-55c1b550eaac)

![image](https://github.com/10824209minggui/Finalreport/assets/149359020/b5d06eef-c5af-4294-84c3-4e3e31e349b7)
