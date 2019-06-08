import tensorflow as tf 

print("Hi!, Current Tensorflow version:")

print(tf.__version__)

def LeNet(img_rows,img_cols,nb_classes, channels):

	model = Sequential(name='LeNet')
    '''
        channels: Specify if the image is grayscale (1) or RGB (3)
        nb_epoch: Number of epochs
        batch_size: Batch size for the model
        nb_classes: Number of classes for classification
    '''

    model = keras.Sequential()

    model.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(32,32,1)))
    model.add(layers.AveragePooling2D())

    model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(layers.AveragePooling2D())

    model.add(layers.Flatten())

    model.add(layers.Dense(units=120, activation='relu'))

    model.add(layers.Dense(units=84, activation='relu'))

    model.add(layers.Dense(nb_classes=10, activation = 'softmax'))

    return model


def Alexnet(img_rows,img_cols,nb_classes, channels):

	'''
        channels: Specify if the image is grayscale (1) or RGB (3)
        nb_epoch: Number of epochs
        batch_size: Batch size for the model
        nb_classes: Number of classes for classification
    '''

    model = Sequential(name='Alexnet')

    #-----------------------------------------------------------------------------------------------
    # Layer 1: 224×224×3 input image with 96 kernels of size 11×11×3 with a stride of 4 pixels    model.add(tf.keras.layers.Conv2D(64, (channels, channels), padding='same', activation='relu', input_shape=input_shape, name='block1_conv1'))
    
    # 1st Convolutional Layer
    model.add(tf.keras.layers.Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11), strides=(4,4), padding=’valid’))
    model.add(tf.keras.layers.Activation(‘relu’)) 
    # Max Pooling
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding=’valid’))

    # 2nd Convolutional Layer 
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding=’valid’))
    model.add(tf.keras.layers.Activation(‘relu’))
    # Max Pooling
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding=’valid’))

    # 3rd Convolutional Layer
    model.add(tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding=’valid’))
    model.add(tf.keras.layers.Activation(‘relu’))

    # 4th Convolutional Layer
    model.add(tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding=’valid’))
    model.add(tf.keras.layers.Activation(‘relu’))

    # 5th Convolutional Layer
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding=’valid’))
    model.add(Activation(‘relu’))
    # Max Pooling
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding=’valid’))

    # Passing it to a Fully Connected layer
    model.add(tf.keras.layers.Flatten())
    # 1st Fully Connected Layer
    model.add(tf.keras.layers.Dense(4096, input_shape=(224*224*3,)))
    model.add(tf.keras.layers.Activation(‘relu’))
    # Add Dropout to prevent overfitting
    model.add(tf.keras.layers.Dropout(0.4))

    # 2nd Fully Connected Layer
    model.add(tf.keras.layers.Dense(4096))
    model.add(tf.keras.layers.Activation(‘relu’))
    # Add Dropout
    model.add(tf.keras.layers.Dropout(0.4))

    # 3rd Fully Connected Layer
    model.add(tf.keras.layers.Dense(1000))
    model.add(tf.keras.layers.Activation(‘relu’))
    # Add Dropout
    model.add(tf.keras.layers.Dropout(0.4))

   # Output Layer
   model.add(tf.keras.layers.Dense(17))
   model.add(tf.keras.layers.Activation(‘softmax’))

   model.summary()


def VGG(img_rows,img_cols,nb_classes, channels):
    '''
        channels: Specify if the image is grayscale (1) or RGB (3)
        nb_epoch: Number of epochs
        batch_size: Batch size for the model
        nb_classes: Number of classes for classification
    '''
    
    input_shape = (img_rows, img_cols, channels) #  (height, width, channel RGB)

    #
    model = Sequential(name='vgg16')

    # block1
    model.add(tf.keras.layers.Conv2D(64, (channels, channels), padding='same', activation='relu', input_shape=input_shape, name='block1_conv1'))
    model.add(tf.keras.layers.Conv2D(64, (channels, channels), padding='same', activation='relu', name='block1_conv2'))
    model.add(tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), name='block1_pool'))
    model.add(tf.keras.layers.BatchNormalization())


    # block2
    model.add(tf.keras.layers.Conv2D(128, (channels, channels), padding='same', activation='relu', name='block2_conv1'))
    model.add(tf.keras.layers.Conv2D(128, (channels, channels), padding='same', activation='relu', name='block2_conv2'))
    model.add(tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), name='block2_pool'))
    model.add(tf.keras.layers.BatchNormalization())

    # block3
    model.add(tf.keras.layers.Conv2D(256, (channels, channels), padding='same', activation='relu', name='block3_conv1'))
    model.add(tf.keras.layers.Conv2D(256, (channels, channels), padding='same', activation='relu', name='block3_conv2'))
    model.add(tf.keras.layers.Conv2D(256, (channels, channels), padding='same', activation='relu', name='block3_conv3'))
    model.add(tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), name='block3_pool'))
    model.add(tf.keras.layers.BatchNormalization())

    # block4
    model.add(tf.keras.layers.Conv2D(512, (channels, channels), padding='same', activation='relu', name='block4_conv1'))
    model.add(tf.keras.layers.Conv2D(512, (channels, channels), padding='same', activation='relu', name='block4_conv2'))
    model.add(tf.keras.layers.Conv2D(512, (channels, channels), padding='same', activation='relu', name='block4_conv3'))
    model.add(tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), name='block4_pool'))
    model.add(tf.keras.layers.BatchNormalization())

    # block5
    model.add(tf.keras.layers.Conv2D(512, (channels, channels), padding='same', activation='relu', name='block5_conv1'))
    model.add(tf.keras.layers.Conv2D(512, (channels, channels), padding='same', activation='relu', name='block5_conv2'))
    model.add(tf.keras.layers.Conv2D(512, (channels, channels), padding='same', activation='relu', name='block5_conv3'))
    model.add(tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), name='block5_pool'))
    model.add(tf.keras.layers.BatchNormalization())
    

    # Classification
    model.add(tf.keras.layers.Flatten(name='flatten'))
    model.add(tf.keras.layers.Dense(4096, activation='relu', name='fully_c1'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(2048, activation='relu', name='fully_c2'))
    model.add(tf.keras.layers.Dense(nb_classes, activation='softmax', name='Predicted_classes'))

    # show me the network!!!
    model.summary()
  
    return model





