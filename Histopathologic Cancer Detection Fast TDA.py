import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import glob

import pandas as pd
import csv

# data = pd.read_csv('TrainImgsBnW.csv', low_memory=False)
# data = pd.read_csv('TrainImgsBnW.csv', low_memory=False)
# print(data)
# exit()
##############################################################################
##############################################################################
##############################################################################
##############################################################################

# from kmapper import jupyter
import kmapper as km
import numpy as np

import io
import sys
import base64

import numpy as np
import sklearn
import kmapper as km
import pandas as pd

# try:
#     from scipy.misc import imsave, toimage
# except ImportError as e:
#     print("imsave requires you to install pillow. Run `pip install pillow` and then try again.")
#     sys.exit()

from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import Isomap
from sklearn.preprocessing import MinMaxScaler

############################################################################
#Preparing Data#############################################################
############################################################################

print('***Start***')

# iStart = 0
# jStart = 0
# steps = 96/12

# iEnd = steps
# jEnd = steps

# imgs = []

# Img = cv2.imread('Extra Data/train/53b0db35f042759861597b2a48b8e7cd9c6c9192.tif')[:,:,0]
# print(Img)
# print(Img.shape)
# # print(Img[0:7,0:7])
# while jStart<Img.shape[0]:
#     while iStart<Img.shape[1]:
#         # print(jStart,jEnd,iStart,iEnd)
#         # print(Img[int(jStart):int(jEnd),int(iStart):int(iEnd)])
#         # print()
#         part = Img[int(jStart):int(jEnd),int(iStart):int(iEnd)]
#         row_data = part.ravel()
#         imgs.append(row_data)

#         iStart += steps
#         iEnd += steps
#     iStart = 0
#     iEnd = steps
#     jStart += steps
#     jEnd += steps

# imgs = np.asarray(imgs)
# df = pd.DataFrame(imgs)
# df.to_csv('oneImg.csv')

# print('End oneImg.csv')

# dataset = pd.read_csv('oneImg.csv')
# X = dataset.iloc[:,1:].values

# index = 1000
dataset = pd.read_csv('TrainImgsBnW - With Label.csv')
X = dataset.iloc[:1000,3:].values

# y = pd.read_csv('TrainImgsBnW - With Label.csv')
y = dataset.iloc[:1000,1].values
dataset = 0
print(X)
exit()
# print("SAMPLE",X[0])
# print("SHAPE",X.shape)

# print("SAMPLE",y[0])
# print("SHAPE",y.shape)

mapper = km.KeplerMapper(verbose=2)

projected_X = mapper.fit_transform(X,
    projection=sklearn.manifold.TSNE(),
    scaler=[None, None, MinMaxScaler()])

# # print(projected_X)
# # print("SHAPE",projected_X.shape)

np.save('projected_X', projected_X)
print('***projected_X Saved***')
# projected_X = np.load(outfile + '.npy')

from sklearn import cluster
graph = mapper.map(projected_X,
                   clusterer=cluster.AgglomerativeClustering(n_clusters=2,
                                                             linkage="complete",
                                                             affinity="cosine"),
                   overlap_perc=0.2)

print(graph)
np.save('graph', projected_X)
print('***graph Saved***')

# count = -1
# tooltip_s = []
# for i in y:
#     count += 1
#     if i == 0:
#         img_tag = """
#                     <div style="position: relative; top: 0; left: 1px; font-size:25px">%s(%s)</div></div>
#                 """%(count, i)
#     else:
#         img_tag = """
#                     <div style="position: relative; top: 0; left: 1px; font-size:25px; color:red">%s(%s)</div></div>
#                 """%(count, i) 
#     tooltip_s.append(img_tag)
# tooltip_s = np.array(tooltip_s)

# html = mapper.visualize(graph,
#                         # X=interpretable_inverse_X,
#                         # X_names=interpretable_inverse_X_names,
#                         path_html="Histopathologic Cancer Detection.html",
#                         lens=projected_X,
#                         lens_names=["ISOMAP1", "ISOMAP2"],
#                         title="Histopathologic Cancer Detection",
#                         custom_tooltips=tooltip_s,
#                         color_function=y
#                         )


html = mapper.visualize(graph,
                        path_html="Histopathologic Cancer Detection - No Labels.html",
                        lens=projected_X,
                        lens_names=["ISOMAP1", "ISOMAP2"],
                        title="Histopathologic Cancer Detection - No Labels"
                        )