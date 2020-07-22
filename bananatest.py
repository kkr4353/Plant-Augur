
import numpy as np
import cv2
import pickle

#############################################

frameWidth= 640         # CAMERA RESOLUTION
frameHeight = 480
brightness = 90#180
threshold = 0.60     # PROBABLITY THRESHOLD#0.75 0.08 smooth //0.80
font = cv2.FONT_HERSHEY_SIMPLEX
##############################################

# SETUP THE VIDEO CAMERA
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)
# IMPORT THE TRANNIED MODEL
pickle_in = open("fourth.p", "rb")  ##rb = READ BYTE
model = pickle.load(pickle_in)

def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img =cv2.equalizeHist(img)
    return img
def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img
def getCalssName(classNo):
    if   classNo == 0: return 'unripe'
    elif classNo == 1: return 'ripe'
    elif classNo == 2: return 'Overripe'


while True:

    # READ IMAGE
    success, imgOrignal = cap.read()
    # PROCESS IMAGE
    img = np.asarray(imgOrignal)
    img = cv2.resize(img, (32,32))
    img = preprocessing(img)
    cv2.imshow("Processed Image", img)
    img = img.reshape(1,32, 32, 1)
    cv2.putText(imgOrignal, "CLASS: " , (20, 35), font, 0.75, (0,255, 0), 2, cv2.QT_FONT_NORMAL) #QT_FONT_DEMIBOLD
    cv2.putText(imgOrignal, "PROBABILITY: ", (300, 35), font, 0.75, (0, 255, 0),2, cv2.QT_FONT_NORMAL)
    # PREDICT IMAGE
    predictions = model.predict(img)
    classIndex = model.predict_classes(img)
    probabilityValue =np.amax(predictions)
    if probabilityValue > threshold:
        #print(getCalssName(classIndex))
        cv2.putText(imgOrignal,str(classIndex)+" "+str(getCalssName(classIndex)), (120, 35), font, 0.75, (0, 255, 0), 2, cv2.QT_FONT_NORMAL)
        cv2.putText(imgOrignal, str(round(probabilityValue*100,2) )+"%", (460, 35), font, 0.75, (0, 255, 0), 2, cv2.QT_FONT_NORMAL)
    cv2.imshow("Result", imgOrignal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break