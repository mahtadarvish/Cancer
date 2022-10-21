import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import glob

import pandas as pd
import csv

# cv2.imshow('org', cv2.imread('Extra Data/train/00ad141a73aee891a8fc966306a72248416ffbfd.tif'))
# cv2.imshow('bnw', cv2.imread('train-BnW/00ad141a73aee891a8fc966306a72248416ffbfd.tif'))
# cv2.imshow('R', cv2.imread('train-BnW-BoldRed/00ad141a73aee891a8fc966306a72248416ffbfd.tif'))
# cv2.waitKey(0)
# exit()
# Read Images & Convert To Black & White
count = 0
for image in glob.glob('Extra Data/test/*.tif'):
    count += 1
    print(count)
    I = cv2.imread(image)
    IGray = (
    			(
    				(I[:,:,0]).astype(np.float64) +
    				(I[:,:,1]).astype(np.float64) +
    				(I[:,:,2]).astype(np.float64)
    			)//3
    		).astype(np.uint8)
    cv2.imwrite('test-BnW/'+image.split('\\')[1], IGray)

    # img = Image.open(image)
    # imggray = img.convert('LA')
    # imgmat = np.array( list(imggray.getdata(band = 0)), float)
    # imgmat.shape = (imggray.size[1], imggray.size[0])
    # imgmat = np.matrix(imgmat)

    # U, S, Vt = np.linalg.svd(imgmat) #single value decomposition

    # cmpimg = np.matrix(U[:, :50]) * np.diag(S[:50]) * np.matrix(Vt[:50,:])
    # result = Image.fromarray((cmpimg ).astype(np.uint8))
    # result.save('train/'+image.split('\\')[1])
    # if count>=5000:
    #     break
print('End train-BnW')
count = 0
exit()

##############################################################################
##############################################################################
##############################################################################
##############################################################################

# # Read Images & Convert To Vector (.CSV File)
# count =0
# imgs = []
# for image in glob.glob('train-BnW/*.tif'):
#     count += 1
#     print(count)
#     Img = cv2.imread(image)[:,:,0]
#     Img = cv2.resize(Img, None, fx = 0.25, fy = 0.25, interpolation = cv2.INTER_CUBIC)
#     row_data = Img.ravel()

#     imgName = (image.split('\\')[1]).split('.')[0]

#     # imgs.append(row_data)
#     imgs.append(np.append(imgName, row_data))
#     # df = pd.DataFrame([row_data])
#     # df.to_csv('TestImgs2.csv', mode = 'a', header = False)

# imgs = np.asarray(imgs)
# df = pd.DataFrame(imgs)
# df.to_csv('TrainImgsBnW.csv')

# print('End TrainImgsBnW.csv')
# count =0
# imgs = []

##############################################################################
##############################################################################
##############################################################################
##############################################################################

count =0
firstHeader = True

imgs = []
for image in glob.glob('train-BnW/*.tif'):
    count += 1
    print(count)
    Img = cv2.imread(image)[:,:,0]
    Img = cv2.resize(Img, None, fx = 0.25, fy = 0.25, interpolation = cv2.INTER_CUBIC)

    row_data = Img.ravel()

    imgName = (image.split('\\')[1]).split('.')[0]
    imgs.append(np.append(imgName, row_data))

    if count == 1000:
        print('print')
        if firstHeader == True:

            imgs = np.asarray(imgs)
            df = pd.DataFrame(imgs)
            df.to_csv('TrainImgsBnW.csv')

            firstHeader = False
            imgs = []
            count = 0
        
        else:

            imgs = np.asarray(imgs)
            df = pd.DataFrame(imgs)
            df.to_csv('TrainImgsBnW.csv', mode = 'a', header = False)
            imgs = []
            count = 0


imgs = np.asarray(imgs)
df = pd.DataFrame(imgs)
df.to_csv('TrainImgsBnW.csv', mode = 'a', header = False)
imgs = []
count = 0
exit()