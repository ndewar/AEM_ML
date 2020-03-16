import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import keras
import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

import sklearn
import pandas
import pickle
import pydot

# import the data
data = np.load('data_nadj4_stdTrue.npy')
dataLabels = np.load('labels_nadj4_stdTrue.npy')

# format the labels
temp = np.zeros((len(dataLabels),1))
temp[:,0] = dataLabels
dataLabels = temp

# reshuffle the vectors
data_preped=np.transpose(data, (0, 2, 3, 1))

# set large values to mean of that dimension
for k in range(0,37):
    for m in range(0,4):
        
        # get the current timegate
        currData=data_preped[:,:,k,m]

        # set values over 5000 to the mean of the timegate
        currData[np.abs(currData)>5000]=np.mean(currData[np.abs(currData)<5000])

        # do the same to nans
        currData=np.nan_to_num(currData, nan=np.mean(currData[np.abs(currData)<5000]))

        # put it back
        data_preped[:,:,k,m]=currData
        
# set nans to 0 (there shouldnt be any but who knows)
X=np.nan_to_num(data_preped)

# scale each timegate to be between -1 and 1
min_max_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
for i in range(np.shape(X)[-1]):
    for j in range(np.shape(X)[1]):
        
        # do it for all examples, for this moment
        X[:,j,:,i] = min_max_scaler.fit_transform(X[:,j,:,i])
        
        # for the timegates where the LM never exists, set it back to 0 so it sits in the middle of the scaled range
        for k in range(np.shape(X)[2]):
            if i==1 or i==3:
                if k>=28:
                    X[:,j,k,i]=X[:,j,k,i]-X[:,j,k,i]

# make some random indices, then take 20000 examples for training and the rest for test, results in about 85/15 split
indices=np.random.permutation(X.shape[0])
X_train=X[indices[:20000],:,:,:]
X_test=X[indices[20000:],:,:,:]
Y_train=dataLabels[indices[:20000],:]
Y_test=dataLabels[indices[20000:],:]

def model(input_shape):

    """
    input_shape: The height, width and channels as a tuple.  
        Note that this does not include the 'batch' as a dimension.
        If you have a batch like 'X_train', 
        then you can provide the input_shape using
        X_train.shape[1:]
    """

    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    #X = ZeroPadding2D((3, 3))(X_input)
    
    # initializer to use
    initToUse=keras.initializers.glorot_normal(seed=0)
    #initToUse=keras.initializers.he_normal(seed=0)
    
    # for the leakly relu
    alphaParam=0.3

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(16, (2, 3), strides = (1, 1), kernel_initializer=initToUse, name = 'conv0', padding='same')(X_input)
    X = MaxPooling2D((1, 2), name='max_pool0')(X)
    X = keras.activations.relu(X,alpha=alphaParam)
    X = Conv2D(32, (2, 5), strides = (1, 1), kernel_initializer=initToUse, name = 'conv1', padding='same')(X)
    X = MaxPooling2D((1, 2), name='max_pool1')(X)
    X = keras.activations.relu(X,alpha=alphaParam)
    X = Conv2D(64, (2, 7), strides = (1, 1), kernel_initializer=initToUse, name = 'conv2', padding='same')(X)
    X = MaxPooling2D((1, 2), name='max_pool2')(X)
    X = keras.activations.relu(X,alpha=alphaParam)
    X = Conv2D(128, (2, 9), strides = (1, 1), kernel_initializer=initToUse, name = 'conv3', padding='same')(X)
    X = MaxPooling2D((1, 2), name='max_pool3')(X)
    X = keras.activations.relu(X,alpha=alphaParam)
    X = Conv2D(128, (2, 11), strides = (1, 1), kernel_initializer=initToUse, name = 'conv4', padding='same')(X)
    #X = MaxPooling2D((1, 2), name='max_pool3')(X)
    X = keras.activations.relu(X,alpha=alphaParam)
    
    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool_final')(X)
    
    # make another initializer
    initToUse2=keras.initializers.he_normal(seed=0)
    
    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(200, activation='relu', kernel_initializer=initToUse2,name='fc0')(X)
    X = Dense(100, activation='relu', kernel_initializer=initToUse2,name='fc1')(X)
    X = Dense(50, activation='relu', kernel_initializer=initToUse2,name='fc2')(X)
    X = Dense(1, activation='sigmoid', kernel_initializer=initToUse,name='fc3')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='HappyModel')
    ### END CODE HERE ###
    
    return model


def modelv2(inputShape,L2param,alpha):

    model = keras.models.Sequential()
    
    # make the initializer
    initToUse=keras.initializers.glorot_normal(seed=1)
    initToUse2=keras.initializers.he_normal(seed=1)

    # for the leakly relu
    alphaParam=alpha
    
    # short dim
    shortDim=3
    
    # add some regularization to the dense layers
    denseReg=keras.regularizers.l2(L2param)
    
    # add some conv layers
    model.add(Conv2D(16, (shortDim, 3), input_shape=inputShape, kernel_initializer=initToUse, padding='same'))
    model.add(MaxPooling2D(pool_size=(1,2), strides=None))
    model.add(keras.layers.LeakyReLU(alpha=alphaParam))
    
    model.add(Conv2D(32, (shortDim, 5), kernel_initializer=initToUse, padding='same'))
    model.add(MaxPooling2D(pool_size=(1,2), strides=None))
    model.add(keras.layers.LeakyReLU(alpha=alphaParam))
    
    model.add(Conv2D(64, (shortDim, 7), kernel_initializer=initToUse, padding='same'))
    model.add(MaxPooling2D(pool_size=(1,2), strides=None))
    model.add(keras.layers.LeakyReLU(alpha=alphaParam))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128, (shortDim, 9), kernel_initializer=initToUse, padding='same'))
    model.add(MaxPooling2D(pool_size=(1,2), strides=None))
    model.add(keras.layers.LeakyReLU(alpha=alphaParam))
    
    model.add(Conv2D(256, (shortDim, 11), kernel_initializer=initToUse, padding='same'))
    model.add(MaxPooling2D(pool_size=(1,2), strides=None))
    model.add(keras.layers.LeakyReLU(alpha=alphaParam))
    
    # flatten and do some dense layers
    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_regularizer=denseReg)) 
    model.add(Dense(160, activation='relu', kernel_regularizer=denseReg))
    model.add(Dense(96, activation='relu', kernel_regularizer=denseReg))
    model.add(Dense(32, activation='relu', kernel_regularizer=denseReg))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    return model

def modelv3(inputShape,L2,alpha):

    model = keras.models.Sequential()
    
    # make the initializer
    initToUse=keras.initializers.glorot_normal(seed=1)
    initToUse2=keras.initializers.he_normal(seed=1)
    
    # short dim
    shortDim=3
    
    # add some regularization to the dense layers
    denseReg=keras.regularizers.l2(L2)
    
    # add some conv layers
    model.add(Conv2D(16, (shortDim, 3), input_shape=inputShape, kernel_initializer=initToUse, padding='same'))
    model.add(MaxPooling2D(pool_size=(1,2), strides=None))
    model.add(keras.layers.LeakyReLU(alpha=alpha))
    
    model.add(Conv2D(32, (shortDim, 5), kernel_initializer=initToUse, padding='same'))
    model.add(MaxPooling2D(pool_size=(1,2), strides=None))
    model.add(keras.layers.LeakyReLU(alpha=alpha))
    
    model.add(Conv2D(64, (shortDim, 7), kernel_initializer=initToUse, padding='same'))
    model.add(MaxPooling2D(pool_size=(1,2), strides=None))
    model.add(keras.layers.LeakyReLU(alpha=alpha))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128, (shortDim, 9), kernel_initializer=initToUse, padding='same'))
    model.add(MaxPooling2D(pool_size=(1,2), strides=None))
    model.add(keras.layers.LeakyReLU(alpha=alpha))
    
    model.add(Conv2D(256, (shortDim, 11), kernel_initializer=initToUse, padding='same'))
    #model.add(MaxPooling2D(pool_size=(1,2), strides=None))
    model.add(keras.layers.LeakyReLU(alpha=alpha))
    
    model.add(Conv2D(256, (shortDim, 11), kernel_initializer=initToUse, padding='same'))
    model.add(MaxPooling2D(pool_size=(1,2), strides=None))
    model.add(keras.layers.LeakyReLU(alpha=alpha))
    
    # flatten and do some dense layers
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_regularizer=denseReg)) 
    model.add(Dense(256, activation='relu', kernel_regularizer=denseReg)) 
    model.add(Dense(160, activation='relu', kernel_regularizer=denseReg))
    model.add(Dense(96, activation='relu', kernel_regularizer=denseReg))
    model.add(Dense(32, activation='relu', kernel_regularizer=denseReg))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    return model

def trainEnsemble(modelType,ensembleSize,X_train,currLearingRate,currEpochNum,currL2param,currBatchSize,trialEpochNum,currAlphaParam,verbFlag):
    
    # make a list of models and other lists and arrays we need
    test_acc=np.zeros((ensembleSize,1))
    Models=[]
    adam=keras.optimizers.Adam(beta_1=0.9, beta_2=0.999,lr=currLearingRate)
    
    # train the ensemble of models
    for i in range(ensembleSize):
        
        # make the flag for checking if the training got stuck
        whileFlag=1
        while whileFlag==1:

            # make the model based on the model type
            if modelType==1:
                tempModel = modelv2(X_train.shape[1:],currL2param,currAlphaParam)
            elif modelType==2:
                tempModel = modelv3(X_train.shape[1:],currL2param,currAlphaParam)
                

            # compile the model
            tempModel.compile(optimizer = adam, loss = "binary_crossentropy", metrics = ["accuracy"])

            # fit the model
            tempModel.fit(x = X_train, y = Y_train, epochs = trialEpochNum, batch_size = currBatchSize,  verbose=verbFlag)

            # test to see if it was stuck
            temp=tempModel.evaluate(x = X_train, y = Y_train)

            if (temp[1]*100)>73:

                # train for all the epochs
                tempModel.fit(x = X_train, y = Y_train, epochs = currEpochNum, batch_size = currBatchSize,  verbose=verbFlag)

                # predict and store the results
                temp=tempModel.evaluate(x = X_test, y = Y_test)
                test_acc[i]=temp[1]

                # append the model and end the while loop
                Models.append(tempModel)
                whileFlag=0

    test_acc_var=np.var(100*test_acc)

    # get the mean label, round it and find the emsemble test acc
    meanLabels=(Models[0].predict(X_test)+Models[1].predict(X_test)+Models[2].predict(X_test)+Models[3].predict(X_test)+Models[4].predict(X_test))/5
    meanLabels=np.round(meanLabels)
    wrongLabels=Y_test-meanLabels
    test_acc_mean=1-np.sum(np.abs(wrongLabels))/len(Y_test)

    print("\n Final ensemble test accuracy of: ",str(np.round(test_acc_mean*100,2)),", mean test accuracy of:",np.str(np.round(np.mean(test_acc)*100,2)),"with a std of: ",str(np.round(test_acc_var,2)))

    return Models, test_acc_mean, test_acc_var, test_acc

def reset_keras():
    
    # if the stuff below doesnt work do this
    K.clear_session()
    
    sess = K.get_session()
    K.clear_session()
    sess.close()
    sess = K.get_session()

    try:
        del classifier # this is from global space - change this as you need
    except:
        pass

    #print(gc.collect()) # if it's done something you should see a number being outputted

    # use the same config as you used to create the session
    config = K.tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    K.set_session(K.tf.Session(config=config))



# set the random seed
np.random.seed(2)

# set the number of stoachastic samples
examplesToRun=100

# set epochs
trialEpochNum=20
epochNumArray=np.random.randint(200, 401, size=(examplesToRun,))

# set the learning rate, l2 param and batch size
learningRateArray=np.power(np.random.randint(1, 10, size=(examplesToRun,)).astype('float64'),-1*np.random.randint(4, 6, size=(examplesToRun,)))
l2paramArray=np.power(np.random.randint(1, 10, size=(examplesToRun,)).astype('float64'),-1*np.random.randint(2, 4, size=(examplesToRun,)))
batchSizeArray=np.random.randint(200, 601, size=(examplesToRun,))  # best was 475 but 480 is a multiple of 32, fits in memory better?

# set the alpha for the leakly relu
alphaParamArray=np.power(np.random.randint(1, 10, size=(examplesToRun,)).astype('float64'),-1*np.random.randint(1, 3, size=(examplesToRun,)))

# make empty arrays
results=[]
test_acc_array=[]
test_acc_mat=[]
#for i in range(examplesToRun):
    
#    # run an ensemble for three different epoch numbers
#    #temp = trainEnsemble(1,5,X_train,learningRateArray[i],epochNumArray[i],l2paramArray[i],batchSizeArray[i],trialEpochNum,alphaParamArray[i],1)
#    temp = trainEnsemble(1,5,X_train,0.0008,epochNumArray[i],l2paramArray[i],batchSizeArray[i],trialEpochNum,alphaParamArray[i],1)
#    #temp = trainEnsemble(2,5,X_train,0.0008,200,0.1,512,trialEpochNum,0.3,1)
#    reset_keras()
#    results.append(temp)
#    test_acc_array.append(temp[1])
#    test_acc_mat.append(temp[3])
    
    
#np.savetxt('test_acc_array.txt', test_acc_array, delimiter=',')
#np.savetxt('test_acc_mat.txt', test_acc_mat, delimiter=',')
np.savetxt('input_params.txt', (epochNumArray,l2paramArray,batchSizeArray,alphaParamArray), delimiter=',')

