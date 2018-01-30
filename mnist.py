# 3. Import libraries and modules
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import math
import time
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

def find_lines(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(gray, kernel=kernel, iterations=1)
    edges = cv2.Canny(erosion, 120, 100, apertureSize=5)
    minLineLength = 100
    maxLineGap = 30
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, np.array([]), minLineLength, maxLineGap)

    realLines = []
    lineEquation = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if not realLines:
                realLines.append(line)
            else:
                flag = 0
                for foundline in realLines:
                    x11 = foundline[0][0]
                    y11 = foundline[0][1]
                    x22 = foundline[0][2]
                    y22 = foundline[0][3]
                    if (abs(x11 - x1) < 30 and abs(y11 - y1) < 30):
                        if (abs(x22 - x2) < 30 and abs(y22 - y2) < 30):
                            flag = 1

                    if (abs(x11 - x2) < 30 and abs(y11 - y2) < 30):
                        if (abs(x22 - x1) < 30 and abs(y22 - y1) < 30):
                            flag = 1
                if flag == 0:
                    realLines.append(line)
    line_cnt = 1
    for line in realLines:
        newElem = {}
        newElem["id"] = line_cnt
        line_cnt = line_cnt + 1
        color = frame[line[0][1],line[0][0]]
        if(color[0]>color[1]):
            newElem["color"] = 'blue'
        else:
            newElem["color"] = 'green'
        x1 = float(line[0][0])
        y1 = float(line[0][1])
        x2 = float(line[0][2])
        y2 = float(line[0][3])
        k = (y2 - y1) / (x2 - x1)
        k = round(k, 2)
        n =  y1 - k*x1
        n = round(n, 2)
        newElem["k"] = k
        newElem["n"] = n
        if(x2>x1):#  HOCU UVEK DA IMAM MANJI X NA X1 A VECI X NA X2 RADI UPOREDJIVANJA , ISTO I ZA Y
            newElem["x1"] = x1
            newElem["x2"] = x2
        else:
            newElem["x1"] = x2
            newElem["x2"] = x1

        if(y2>y1):
            newElem["y1"] = y1
            newElem["y2"] = y2
        else:
            newElem["y1"] = y2
            newElem["y2"] = y1

        lineEquation.append(newElem)

    return lineEquation

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
    if( northDistance!=0 and westDistance!=0):
        for r in xrange(0, image.shape[0]):
            for c in xrange(0, image.shape[1]):
                if(northDistance == 0):
                    image[r - northDistance , c - westDistance + 1] = image[r, c]
                    image[r, c] = 0
                elif(westDistance == 0):
                    image[r - northDistance + 1, c - westDistance ] = image[r, c]
                    image[r, c] = 0
                else:
                    image[r - northDistance + 1, c - westDistance + 1] = image[r, c]
                    image[r, c] = 0

    return image

def next_frame(x1,y1,elements):
    ret_Val = []
    for element in elements:
        x2 = element["x"]
        y2 = element["y"]
        if(abs(math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1)))) < 8:#minimalna distanca
            ret_Val.append(element)
    return ret_Val

def dot_distance(x1,x2,y1,y2):
    d = abs(math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1)))
    return d

def retun_passed_lines(lines,x,y):
    linespassed = []
    for line in lines:
        k = line["k"]
        n = line["n"]
        result = k * x + n - y
        if result < 0:
            linespassed.append(line["id"])

    return linespassed

def return_value(Y_value):
    for i in xrange(0,Y_value.shape[1]):
        if Y_value[0,i] == 1:
            return i

def check_dot(lines,x,y):
    for line in lines:
        k = line["k"]
        n = line["n"]
        result = abs(k*x+n-y)
        if(result < 8):#u pravom zivotu treba = 0, ispunjenje jednacine prave ali odsekao sam delove pa radim aproksimaciju
            if(x>=line["x1"] and x<=line["x2"]):
                if(y>=line["y1"] and y<=line["y2"]):
                    return line["id"]
    return None
def line_cross(lines,xcenter,ycenter):

    x1 = xcenter - 14
    y1 = ycenter - 14
    result1 = check_dot(lines,x1,y1)
    if (result1 is not None):
        return result1

    x2 = xcenter - 14
    y2 = ycenter + 14
    result2 = check_dot(lines,x2,y2)
    if (result2 is not None):
        return result2

    x3 = xcenter + 14
    y3 = ycenter - 14
    result3 = check_dot(lines,x3,y3)
    if (result3 is not None):
        return result3

    x4 = xcenter + 14
    y4 = ycenter + 14
    result4 = check_dot(lines,x4,y4)
    if(result4 is not None):
        return result4

    x5 = xcenter
    y5 = ycenter
    result5 = check_dot(lines, x4, y4)
    if (result5 is not None):
        return result5

    x6 = xcenter + 14
    y6 = ycenter
    result6 = check_dot(lines, x4, y4)
    if (result6 is not None):
        return result6

    x7 = xcenter - 14
    y7 = ycenter
    result7 = check_dot(lines, x4, y4)
    if (result7 is not None):
        return result7

    x8 = xcenter
    y8 = ycenter + 14
    result8 = check_dot(lines, x4, y4)
    if (result8 is not None):
        return result8

    x9 = xcenter
    y9 = ycenter - 14
    result9 = check_dot(lines, x4, y4)
    if (result9 is not None):
        return result9


    '''
    if check_dot(lines, xcenter, ycenter):
        # print ('x1 y1 pipnuo '+str(x1)+' | '+str(y1))
        return True
    '''
    return None

def test_model(model):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    Y_test = np_utils.to_categorical(y_test, 10)
    X_test = compute_dataset(X_test)


    # 10. Evaluate model on test data
    score = model.evaluate(X_test, Y_test, verbose=0)
    print score

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
#test_model(model)
impath = 'C:/Users/Nikola/Desktop/selfi.png'
path = 'C:/Users/Nikola/Desktop/soft-projekat/video-0.avi'
selfi = cv2.imread(impath)

lower = np.array([180, 180, 180])
upper = np.array([255, 255, 255])

lineEquations = []
numbers=[]
number_cnt = 0
result = 0
frame_counter= 0
cap = cv2.VideoCapture(path)
while(True):
    ret, frame = cap.read()
    start_time = time.time()
    frame_counter = frame_counter +1
    if frame_counter == 1:
        lineEquations = find_lines(frame)
        print (lineEquations)

    counter = 0;
    mask = cv2.inRange(frame, lower, upper)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    label_im, nb_labels = ndimage.label(res)
    objects = ndimage.find_objects(label_im)

    for i in range(nb_labels):
        loc = objects[i]
        (xc, yc) = ((loc[1].stop + loc[1].start) / 2,(loc[0].stop + loc[0].start) / 2)#sredina(loc[0].stop + loc[0].start) / 2)
        (dxc, dyc) = (loc[1].stop - loc[1].start,(loc[0].stop - loc[0].start))
        if(yc>14 and yc < frame.shape[0]-14 and xc>14 and xc<frame.shape[1]-14):
            if (dxc > 12 or dyc > 12):
                #cv2.circle(res, (xc, yc), 16, (25, 25, 255), 1)

                candidates = next_frame(xc,yc,numbers)

                if(len(candidates) == 0):

                    maci = res[yc-14:yc+14, xc-14:xc+14]
                    macigray = cv2.cvtColor(maci, cv2.COLOR_BGR2GRAY)
                    maci_computed = compute_image(macigray)
                    maci_test = maci_computed.reshape(1, 1, maci_computed.shape[0], maci_computed.shape[1])
                    prediction = model.predict(maci_test)
                    newelement = {"x":xc,"y":yc,"value":return_value(prediction),"timestamp":start_time,"transform":0}
                    newelement["passed"] = retun_passed_lines(lineEquations,xc,yc)
                    numbers.append(newelement)
                    #print('added'+str(newelement["value"]))

                if(len(candidates) == 1):
                    for number in numbers:
                        if(number == candidates[0]):
                            number["transform"] = number["transform"] + start_time
                            number["x"] = xc
                            number["y"] = yc
                            number["timestamp"] = start_time
                            if number["transform"]>1:
                                maci = res[yc - 14:yc + 14, xc - 14:xc + 14]
                                macigray = cv2.cvtColor(maci, cv2.COLOR_BGR2GRAY)
                                maci_computed = compute_image(macigray)
                                maci_test = maci_computed.reshape(1, 1, maci_computed.shape[0],maci_computed.shape[1])
                                prediction = model.predict(maci_test)
                                number["value"] = return_value(prediction)
                                number["transform"] = 0
                            linepassed = line_cross(lineEquations,xc,yc)
                            if linepassed is not None:
                                flag = False
                                for line in number["passed"]:
                                    if line == linepassed :
                                        flag = True
                                        #print('areadly pasesd')
                                if flag == False:
                                    number["passed"].append(linepassed)
                                    for line in lineEquations:
                                        if line["id"] == linepassed:
                                            if line["color"] == 'blue':
                                                result = result + number["value"]
                                                #print('sabiranje')
                                            else:
                                                result = result - number["value"]
                                                #print('oduzimanje')
                #if(len(candidates)>1):
                    #print'prekpalanja'

                for number in numbers:
                    if start_time - number["timestamp"] > 3:
                        numbers.remove(number)
                        #print 'Izbrisao'+str(number["value"])

    cv2.putText(frame, 'Result: '+ str(result), (95, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 90, 255), 1)
    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.waitKey(0)
cv2.destroyAllWindows()