import numpy as np
import cv2
import pandas
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

import sys # for sys.stdout

labels = 'E:/IP/Labels/shuffled_labels.csv'
npzfile = 'E:/IP/Labels/labels.npz'
#create npz hiden  file 
df = pandas.read_csv(labels)

rows = df.iterrows()

X_temp = []
Y_temp = []

scan_wavelengths = range(0,1000000)  # the variable being varied
nWLs = len(scan_wavelengths)  # how many steps in the loop

# Progress Bar setup:
ProgMax = 20    # number of dots in progress bar
if nWLs<ProgMax:   ProgMax = nWLs   # if less than 20 points in scan, shorten bar
print "|" +    ProgMax*"-"    + "|     MyFunction() progress"
sys.stdout.write('|'); sys.stdout.flush();  # print start of progress bar
nProg = 0   # fraction of progress

for row in rows:
    image = row[1][0]
    img = cv2.imread(image)
    npimg = np.asarray(img)
    npimg = npimg / 255
    (b, g, r)=cv2.split(npimg)
    npimg=cv2.merge([r,g,b])
    imageClass = row[1][1]
    X_temp.append(npimg)
    Y_temp.append(imageClass)

    # update progress bar:
    if ( row >= nProg*nWLs/ProgMax ):
        '''Print dot at some fraction of the loop.'''
        sys.stdout.write('*'); sys.stdout.flush();  
        nProg = nProg+1
    if ( row >= nWLs-1 ):
        '''If done, write the end of the progress bar'''
        sys.stdout.write('|     done  \n'); sys.stdout.flush(); 
       
# one hot encoding to represent one num as a array like 2= [0 1 0] in a 3 class cnn model     
encoder = LabelEncoder()
encoder.fit(Y_temp)
encoded_Y = encoder.transform(Y_temp)
Y = np_utils.to_categorical(encoded_Y)

np.savez(npzfile, X_train=X_temp,Y_train=Y)

