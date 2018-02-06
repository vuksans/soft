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

def check_dot(line,x,y,mode):
    passedLines = []
    result_threshold = 0
    if mode == 1:
        result_threshold = 5
    if mode == 2:
        result_threshold = 15
    k = line["k"]
    n = line["n"]
    result = abs(k*x+n-y)

    if(result < result_threshold):#u pravom zivotu treba = 0, ispunjenje jednacine prave ali odsekao sam delove pa radim aproksimaciju
        if(x>=line["x1"] and x<=line["x2"]):
            if(y>=line["y1"] and y<=line["y2"]):
                return line["id"]

    return None

def line_cross(lines,xcenter,ycenter,mode,dxc,dyc):

    returnLines = []
    xp = dxc /2
    yp = dyc /2

    x1 = xcenter - xp
    y1 = ycenter - yp
    for line in lines:
        result1 = check_dot(line,x1,y1,mode)
        if (result1 is not None):
            if result1 not in returnLines:
                returnLines.append(result1)

    x2 = xcenter - xp
    y2 = ycenter + yp
    for line in lines:
        result2 = check_dot(line,x2,y2,mode)
        if (result2 is not None):
            if result2 not in returnLines:
                returnLines.append(result2)

    x3 = xcenter + xp
    y3 = ycenter - yp
    for line in lines:
        result3 = check_dot(line,x3,y3,mode)
        if (result3 is not None):
            if result3 not in returnLines:
                returnLines.append(result3)

    x4 = xcenter + xp
    y4 = ycenter + yp
    for line in lines:
        result4 = check_dot(line,x4,y4,mode)
        if(result4 is not None):
            if result4 not in returnLines:
                returnLines.append(result4)

    x5 = xcenter
    y5 = ycenter
    for line in lines:
        result5 = check_dot(line, x5, y5,mode)
        if (result5 is not None):
            if result5 not in returnLines:
                returnLines.append(result5)

    x6 = xcenter + xp
    y6 = ycenter
    for line in lines:
        result6 = check_dot(line, x6, y6,mode)
        if (result6 is not None):
            if result6 not in returnLines:
                returnLines.append(result6)

    x7 = xcenter - xp
    y7 = ycenter
    for line in lines:
        result7 = check_dot(line, x7, y7,mode)
        if (result7 is not None):
            if result7 not in returnLines:
                returnLines.append(result7)

    x8 = xcenter
    y8 = ycenter + yp
    for line in lines:
        result8 = check_dot(line, x8, y8,mode)
        if (result8 is not None):
            if result8 not in returnLines:
                returnLines.append(result8)

    x9 = xcenter
    y9 = ycenter - yp
    for line in lines:
        result9 = check_dot(line, x9, y9,mode)
        if (result9 is not None):
            if result9 not in returnLines:
                returnLines.append(result9)


    '''
    if check_dot(lines, xcenter, ycenter):
        # print ('x1 y1 pipnuo '+str(x1)+' | '+str(y1))
        return True
    '''
    if len(returnLines) == 0:
        return None
    else:
        return returnLines

def get_parnet(number,numbers):
    lowestDistance = 0;
    returnID = 0
    for n in numbers:
        if number["id"] != n["id"]:
            distance = dot_distance(number["x"],n["x"],number["y"],n["y"])
            if distance > 20:
                if lowestDistance == 0:
                    lowestDistance = distance
                    returnID = n["id"]
                elif distance < lowestDistance:
                    lowestDistance = distance
                    returnID = n["id"]

    return {"id":returnID,"distance":lowestDistance}

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

oldFrames =[]
lineEquations = []
numbers=[]
number_cnt = 0
result = 0
frame_counter= 0
cap = cv2.VideoCapture(path)
numberID = 1
vuksa = False
while(True):
    ret, frame = cap.read()
    vuksa = False
    start_time = time.time()
    frame_counter = frame_counter +1
    if frame_counter == 1:
        lineEquations = find_lines(frame)
        print (lineEquations)

    if frame_counter > 1:
        oldFrames = numbers

    counter = 0;
    mask = cv2.inRange(frame, lower, upper)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    label_im, nb_labels = ndimage.label(res)
    objects = ndimage.find_objects(label_im)

    k=0
    for i in range(nb_labels):
        loc = objects[i]
        (xc, yc) = ((loc[1].stop + loc[1].start) / 2,(loc[0].stop + loc[0].start) / 2)#sredina(loc[0].stop + loc[0].start) / 2)
        (dxc, dyc) = (loc[1].stop - loc[1].start,(loc[0].stop - loc[0].start))
        if(yc>14 and yc < frame.shape[0]-14 and xc>14 and xc<frame.shape[1]-14):
            if (dxc > 12 or dyc > 12):
                k = k+1
                cv2.circle(frame, (xc, yc), 2, (25, 25, 255), 2)

                candidates = next_frame(xc,yc,numbers)
                if(len(candidates) == 0):

                    if line_cross(lineEquations,xc,yc,2,dxc,dyc) is None:
                        notNumberFlag = False
                        maci = res[yc-14:yc+14, xc-14:xc+14]
                        macigray = cv2.cvtColor(maci, cv2.COLOR_BGR2GRAY)
                        maci_computed = compute_image(macigray)
                        maci_test = maci_computed.reshape(1, 1, maci_computed.shape[0], maci_computed.shape[1])
                        prediction = model.predict(maci_test)
                        if(return_value(prediction) != None):
                            #print prediction
                            '''
                            if(return_value(prediction) == 3):
                                cv2.imshow('hiii',maci)
                            '''
                            #print model.evaluate(maci_test,prediction,10)
                            newelement = {"id":numberID,"x":xc,"y":yc,"value":return_value(prediction),"timestamp":start_time,"transform":0,"myFrame":frame_counter,"originalValue":return_value(prediction),
                                          "dxc":dxc,"dyc":dyc}
                            newelement["closest"] = {}
                            newelement["covered"] = []
                            newelement["passed"] = retun_passed_lines(lineEquations,xc,yc)

                            '''
                            #funckija preklapanja
                            for number in numbers:
                                if dot_distance(number["x"],newelement["x"],number["y"],newelement["y"]) < 28:
                                    #vuksa = True
                                    #cv2.rectangle(frame, (xc - 14, yc - 14), (xc + 14, yc + 14), (255, 255, 0), 2)
                                    if len(number["covered"]) > 0:
                                        temp = number["covered"][0]
                                        number["covered"].remove(temp)
                                        number["value"] -= temp["value"]
                                        newelement["originalValue"] = temp["value"]
                                        newelement["value"] = temp["value"]
                            '''

                            numbers.append(newelement)
                            numberID = numberID + 1
                            print('added'+str(newelement["value"]))
                            vuksa = True
                            cv2.rectangle(frame,(xc-14,yc-14),(xc+14,yc+14),(255,255,0),2)

                if(len(candidates) > 0):
                    for number in numbers:
                        if(number == candidates[0]):
                            number["transform"] = number["transform"] + start_time
                            number["x"] = xc
                            number["y"] = yc
                            number["timestamp"] = start_time
                            number["myFrame"] = frame_counter
                            '''
                            if number["transform"]>3:
                                if line_cross(lineEquations,number["x"],number["y"],1) is None:
                                    maci = res[yc - 14:yc + 14, xc - 14:xc + 14]
                                    macigray = cv2.cvtColor(maci, cv2.COLOR_BGR2GRAY)
                                    maci_computed = compute_image(macigray)
                                    maci_test = maci_computed.reshape(1, 1, maci_computed.shape[0],maci_computed.shape[1])
                                    prediction = model.predict(maci_test)
                                    number["originalValue"] = return_value(prediction)
                                    number["value"] = number["originalValue"]

                                    for covered in number["covered"]:
                                        number["value"] += covered["value"]
                                    number["transform"] = 0
                            '''
                            linepassed = line_cross(lineEquations,xc,yc,1,dxc,dyc)
                            if linepassed is not None:
                                for lineresult in linepassed:
                                    flag = False
                                    for line in number["passed"]:
                                        if line == lineresult :
                                            flag = True
                                            #print('already pasesd')
                                    if flag == False:
                                        number["passed"].append(lineresult)
                                        for line in lineEquations:
                                            if line["id"] == lineresult:
                                                if line["color"] == 'blue':
                                                    result = result + number["value"]
                                                    #print('sabiranje dodato' + str(number["value"]))
                                                else:
                                                    result = result - number["value"]
                                                    #print('oduzimanje oduzeto'+ str(number["value"]))
                #if(len(candidates)>1):
                    #print'prekpalanja'


    for number in numbers:
        newelement["closest"] = get_parnet(number,numbers)

    coveredUp = []
    for number in numbers:
        if frame_counter - number["myFrame"] > 8:
            #pod 1, ako prelazi liniju, brisem
            if line_cross(lineEquations,number["x"],number["y"],2,number["dxc"],number["dyc"]) is not None:
                numbers.remove(number)
                #print 'Obrisao jer je presao liniju'
                break
            if number["x"] > frame.shape[1] - 50 or number["y"] > frame.shape[0] - 50:
                numbers.remove(number)
                #print 'Obrisao jer je ispao iz frejma'
                break
            else:
                #odkloni za preklop
                #coveredUp.append(number)
                numbers.remove(number)
                print 'Obrisao jer je preklopljen'
            #print 'Izbrisao' + str(number["value"])

    for covered in coveredUp:
        for number in numbers:
            if number["id"] == covered["closest"]["id"]:
                number["covered"].append(covered)
                number["value"] += covered["value"]

    cv2.putText(frame, 'Result: '+ str(result), (95, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 90, 255), 1)
    if vuksa:
        print 'chaoo'
    cv2.imshow('frame',frame)
    #cv2.imshow('res',res)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.waitKey(0)
cv2.destroyAllWindows()