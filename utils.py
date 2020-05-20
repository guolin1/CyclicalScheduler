import tensorflow as tf

def simple_cnn(nfilters:[int], inputshape:(int)=None, 
               kernel_sizes:[int]=None, strides:[int]=None, 
               dropout:[float] = None, batchnorm=True):
    '''
    simple CNN defined by activations, kernel sizes, strides, and batchnorm.
    Dropout can be added. Last value in activations should indicate class size 
    as this will be the unit size of the last dense layer.

    inputs:
        nfilters: []. E.g., [32, 64, 5] indicates 3 layers, the first two
            layers are conv layers with 32 and 64 filters each. The last layer
            is a dense layer with softmax activation containing 5 units, thus 
            indicating 5 classes.
        inputshape: (). E.g., (32, 32, 3) indicates the image has 3 channels and
            has size 32 x 32.
        kernel_sizes: []. Specifies the kernel size for each convolutional layer,
            if none, default is set at 3.
        strides: []. Specifies stride size for each conv layer. If none, 2.
        dropout: []. Specifies dropout probability added to the end of each conv
            layer, after batch normalization.
        batchnorm: True. If true, add batch norm after each conv layer. 
    returns:
        keras model.

    Example:
        model = simple_cnn(nfilters = [16,32,10], 
                        inputshape = (32,32,3), # this is the default
                        kernel_sizes = [3,3], # this is the default
                        strides = [2,2], # this is the default
                        dropout = [0.2,0.1], # default is none
                        batchnorm = True # this is the default)
        
        model.summary()
        Model: "sequential_29"
        _________________________________________________________________
        Layer (type)                 Output Shape              Param #   
        =================================================================
        conv2d_68 (Conv2D)           (None, 32, 32, 16)        448       
        _________________________________________________________________
        batch_normalization_85 (Batc (None, 32, 32, 16)        64        
        _________________________________________________________________
        dropout_18 (Dropout)         (None, 32, 32, 16)        0         
        _________________________________________________________________
        conv2d_69 (Conv2D)           (None, 32, 32, 32)        4640      
        _________________________________________________________________
        batch_normalization_86 (Batc (None, 32, 32, 32)        128       
        _________________________________________________________________
        dropout_19 (Dropout)         (None, 32, 32, 32)        0         
        _________________________________________________________________
        average_pooling2d_26 (Averag (None, 16, 16, 32)        0         
        _________________________________________________________________
        flatten_26 (Flatten)         (None, 8192)              0         
        _________________________________________________________________
        dense_26 (Dense)             (None, 10)                81930     
        =================================================================
        Total params: 87,210
        Trainable params: 87,114
        Non-trainable params: 96
        _________________________________________________________________
    
    ## A simple CNN without dropout with inputs of 32x32x3:
        model = simple_cnn([16,32,10],(32,32,3))
    '''
    nlayers = len(nfilters)-1
    if kernel_sizes is None: kernel_sizes = [3] * nlayers
    if strides is None: strides = [2] * nlayers
    if inputshape is None: inputshape = (32,32,3)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=inputshape))
    for i in range(nlayers):
        model.add(tf.keras.layers.Conv2D(nfilters[i],
                                         kernel_sizes[i],
                                         strides[i],
                                         activation='relu',
                                         padding='same'))
        if batchnorm: model.add(tf.keras.layers.BatchNormalization())
        if dropout is not None: model.add(tf.keras.layers.Dropout(dropout[i]))
    model.add(tf.keras.layers.AveragePooling2D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(nfilters[-1], activation='softmax'))
    return model