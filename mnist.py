# 3. Import libraries and modules
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import math
np.random.seed(123)  # for reproducibility

from scipy import ndimage
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import load_model
from keras import backend as K
K.set_image_dim_ordering('th')


def compute_dataset(set):
    numberofImages = set.shape[0]
    for i in xrange(0,numberofImages):
        print(i)
        img = set[i,0]
        northDistance = 0
        westDistance = 0
        for r in xrange(0, img.shape[0]):
            for c in xrange(0, img.shape[1]):
                if img[r, c] > 0.7:
                    if westDistance == 0:
                        westDistance = c
                    elif westDistance > c:
                        westDistance = c
                    if northDistance == 0:
                        northDistance = r
                    elif northDistance > r:
                        northDistance = r
        for r in xrange(0, img.shape[0]):
            for c in xrange(0, img.shape[1]):
                img[r - northDistance + 1, c - westDistance + 1] = img[r, c]
                img[r, c] = 0
                set[i,0]=img

    return set

def compute_image(image):
    northDistance = 0
    westDistance = 0
    for r in xrange(0, image.shape[0]):
        for c in xrange(0, image.shape[1]):
            if image[r, c] > 0.7:
                if westDistance == 0:
                    westDistance = c
                elif westDistance > c:
                    westDistance = c
                if northDistance == 0:
                    northDistance = r
                elif northDistance > r:
                    northDistance = r
    for r in xrange(0, image.shape[0]):
        for c in xrange(0, image.shape[1]):
            image[r - northDistance + 1, c - westDistance + 1] = image[r, c]
            image[r, c] = 0

    return image

def next_frame(x1,y1,elements):
    ret_Val = []
    for element in elements:
        x2 = element["x"]
        y2 = element["y"]
        if(abs(math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1)))) < 8:
            ret_Val.append(element)
    return ret_Val

def return_value(Y_value):
    for i in xrange(0,Y_value.shape[1]):
        if Y_value[0,i] == 1:
            return i

def train_dataset():
    # 4. Load pre-shuffled MNIST data into train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # 5. Preprocess input data
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # 6. Preprocess class labels
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)

    # X_train = compute_dataset(X_train)
    X_test = compute_dataset(X_test)

    model = load_model('soft_model.h5')

    # 7. Define model architecture
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1, 28, 28)))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    '''

    '''
    # 8. Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # 9. Fit model on training data
    model.fit(X_train, Y_train,
              batch_size=32, nb_epoch=10, verbose=1)

    '''
    # 10. Evaluate model on test data
    score = model.evaluate(X_test, Y_test, verbose=0)
    print score
    '''
    return model


model = load_model('soft_model.h5')
impath = 'C:/Users/Nikola/Desktop/selfi.png'
path = 'C:/Users/Nikola/Desktop/soft-projekat/video-0.avi'
selfi = cv2.imread(impath)

lower = np.array([180, 180, 180])
upper = np.array([255, 255, 255])

numbers=[]
number_cnt = 0
result = 0

cap = cv2.VideoCapture(path)
while(True):
    ret, frame = cap.read()
    counter = 0;
    mask = cv2.inRange(frame, lower, upper)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    label_im, nb_labels = ndimage.label(res)
    objects = ndimage.find_objects(label_im)

    for i in range(nb_labels):
        loc = objects[i]
        (xc, yc) = ((loc[0].stop + loc[0].start) / 2,(loc[1].stop + loc[1].start) / 2)#sredina(loc[0].stop + loc[0].start) / 2)
        (dxc, dyc) = (loc[0].stop - loc[0].start,(loc[1].stop - loc[1].start))
        if (dxc > 12 or dyc > 12):
            #cv2.circle(res, (xc, yc), 16, (25, 25, 255), 1)

            candidates = next_frame(xc,yc,numbers)

            if(len(candidates) == 0):
                maci = res[xc - 14:xc + 14, yc - 14:yc + 14]
                macigray = cv2.cvtColor(maci, cv2.COLOR_BGR2GRAY)
                if macigray is not None:
                    if macigray.shape is not None:
                        if (macigray.shape[0] == 28 and macigray.shape[1] == 28):
                            maci_computed = compute_image(macigray)
                            maci_test = maci_computed.reshape(1, 1, maci_computed.shape[0], maci_computed.shape[1])
                            prediction = model.predict(maci_test)
                            newelement = {"x":xc,"y":yc,"value":return_value(prediction)}
                            numbers.append(newelement)
                            print('added')

            if(len(candidates) == 1):
                for number in numbers:
                    if(number == candidates[0]):
                        #x1 = number["x"]
                        #y1 = number["y"]
                        #x2 = xc
                        #y2 = yc
                        number["x"] = xc
                        number["y"] = yc
                        #print("Distance moved: " + str(abs(math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1)))))



    cv2.putText(frame, 'Counter: '+ str(counter), (90, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 90, 90), 1)
    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.waitKey(0)
cv2.destroyAllWindows()