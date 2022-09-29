'''
建立特征脸的图片来自于本数据集的图像，不参与图像收集与识别
4.2 实验部分
'''

import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import os
import random
from sklearn import decomposition

os.chdir('E:\\PriFace')

a1 = str('.\\att_faces\\')
a2 = str('s')
a3 = str('\\*.pgm')

X = []
labels = []
for i in range(2,40):
    # print(i+1)
    b = a1 + a2 + str(i + 1) + a3
    images = glob.glob(b)
    for img in images:
        labels.append(i)
        img = cv2.imread(img, 0)
        temp = np.resize(img, (img.shape[0] * img.shape[1], 1))
        X.append(temp.T)

labels = np.array(labels)
X = np.array(X).squeeze().T

W = []
labels_w = []
for i in range(0, 2):
    # print(i+1)
    b = a1 + a2 + str(i + 1) + a3
    images = glob.glob(b)
    for img in images:
        labels_w.append(i)
        img = cv2.imread(img, 0)
        temp = np.resize(img, (img.shape[0] * img.shape[1], 1))
        W.append(temp.T)

labels_w = np.array(labels_w)
W = np.array(W).squeeze().T

'''
os.chdir('G:\\face recognition\\数据集new_faces实验\\')
p1 = str('.\\new_faces_111\\')
p2 = str('z')
p3 = str('\\*.jpg')
W = []
for i in range(21):
    v = p1 + p2 + str(i + 1) + p3
    images = glob.glob(v)
    for img in images:
        img = cv2.imread(img, 0)
        temp = np.resize(img, (img.shape[0] * img.shape[1], 1))
        W.append(temp.T)
W = np.array(W).squeeze().T
'''

'''
os.chdir('G:\\face recognition\\Dataset_CASIA\\CASIA-FaceV5\\')
q1 = str('.\\CASIA-FaceV5 (000-099)_b\\')
q2 = str('')
q3 = str('\\*.bmp')

W = []
for i in range(100):
    w = q1 + q2 + str('%03d'%(i)) + q3
    images = glob.glob(w)
    for img in images:
        img = cv2.imread(img, 0)
        temp = np.resize(img, (img.shape[0] * img.shape[1], 1))
        W.append(temp.T)
W = np.array(W).squeeze().T
'''

'''
os.chdir('G:\\face recognition\\数据集Japan实验\\jaffeim\\jaffe2\\')
W = []
b = '.\\*.tiff'
images = glob.glob(b)
for img in images:
    img = cv2.imread(img, 0)
    temp = np.resize(img, (img.shape[0] * img.shape[1], 1))
    W.append(temp.T)         
W = np.array(W).squeeze().T
'''

os.chdir('E:\\PriFace')

a1 = str('.\\test_not_in_orl_1\\')
a2 = str('s')
a3 = str('\\*.jpg')
Y = []
labels_y = []
for i in range(80,84):
    # print(i+1)
    b = a1 + a2 + str(i + 1) + a3
    images = glob.glob(b)
    for img in images:
        labels_y.append(i)
        img = cv2.imread(img, 0)
        temp = np.resize(img, (img.shape[0] * img.shape[1], 1))
        Y.append(temp.T)

labels_y = np.array(labels_y)
Y = np.array(Y).squeeze().T

def euclideanDistance(v1, v2):
    """get euclidean distance of 2 vectors"""
    v1, v2 = np.array(v1), np.array(v2)
    return np.sqrt(np.sum(np.square(v1 - v2)))

def Manhattan(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    return (sum(abs(v1 - v2)))

def genPara(d, r):
    a = []
    for i in range(d):
        a.append(random.gauss(0, 1))
    b = random.uniform(0, r)
    return a, b

repeat_times = 10
samples_for_eigenface = 20
samples_for_train = 304
samples_for_test = 76+4

precision = 0
for i in range(repeat_times):  
    index = random.sample(range(0, 20), samples_for_eigenface)
    eigenface_trainset = W[:, index]
    eigenface_trainset = eigenface_trainset.T
    meanFace = np.mean(eigenface_trainset, axis=0)
    pca = decomposition.PCA(n_components=0.8)
    pca.fit(eigenface_trainset)
    pca.explained_variance_ratio_

    index_train =  []
    step=8
    for i in range(38):
        m = (random.sample(range(10 * i, 10 * (i + 1)), step))
        for i in m:
            index_train.append(i)
    trainingset = X[:, index_train]
    trainingset = trainingset.T
    
    testset1 = np.delete(X, index_train, axis=1)
    testset1 = testset1.T
    index_test2 = random.sample(range(0, 4), 4)
    testset2 = Y[:,index_test2]
    testset2 = testset2.T
    testset = np.vstack((testset1,testset2))
    
    labels_train = labels[index_train]
    labels_test1 = np.delete(labels,index_train)
    labels_test2 = labels_y[index_test2]
    labels_test = np.hstack((labels_test1,labels_test2))

    x_train_pca = pca.transform(trainingset)
    x_test1_pca = pca.transform(testset1)
    x_test2_pca = pca.transform(testset2)
    x_test_pca = np.vstack((x_test1_pca,x_test2_pca))
        
    d = len(pca.explained_variance_ratio_)
    L = 38
    r = 1                       
    gen_e2LSH_family_all = []
    for i in range(L):
        gen_e2LSH_family_all.append([ genPara(d,r),genPara(d,r),genPara(d,r),genPara(d,r)
                                     , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)
                                     , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)
                                     , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)
                                     , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)

                                     , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)
                                     , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)
                                     , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)
                                     , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)
                                     , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)

                                     , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)
                                     , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)
                                     , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)
                                     , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)
                                     , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)
                    ])
                                     

     
    list1 = [0 for i in range(L)]
    
    list1[0] = [ genPara(d,r),genPara(d,r),genPara(d,r),genPara(d,r)
                , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)
                                     , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)
                                     , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)
                                     , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)

                                     , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)
                                     , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)
                                     , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)
                                     , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)
                                     , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)

                                     , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)
                                     , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)
                                     , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)
                                     , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)
                                     , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)
                ]
   
    X_h = []
    for i in range(L):
        for j in range(i*8,(i+1)*8):
            hashVals=[]
            for hab in gen_e2LSH_family_all[i]:
                hashVal = (np.inner(hab[0], x_train_pca[j]) + hab[1]) // r
                hashVals.append(hashVal)
            X_h.append(hashVals)
            
    Z_h = []
    for i in range(L):
        for j in range(i*2,(i+1)*2):
            hashVals=[]
            for hab in gen_e2LSH_family_all[i]:
                hashVal = (np.inner(hab[0], x_test1_pca[j]) + hab[1]) // r
                hashVals.append(hashVal)
            Z_h.append(hashVals)
     
    
    gen_e2LSH_family_y = []
    for i in range(4):
        gen_e2LSH_family_y.append([genPara(d,r),genPara(d,r),genPara(d,r),genPara(d,r)
                                   , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)
                                     , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)
                                     , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)
                                     , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)

                                     , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)
                                     , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)
                                     , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)
                                     , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)
                                     , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)

                                     , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)
                                     , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)
                                     , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)
                                     , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)
                                     , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)
                                  ])
        
        
    list_y = [0 for i in range(4)]
    
    list_y[0] = [genPara(d,r),genPara(d,r),genPara(d,r),genPara(d,r)
                 , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)
                                     , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)
                                     , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)
                                     , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)

                                     , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)
                                     , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)
                                     , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)
                                     , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)
                                     , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)

                                       , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)
                                       , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)
                                     , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)
                                     , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)
                                     , genPara(d, r), genPara(d, r), genPara(d, r), genPara(d, r)
                  ]
   
        

    Y_h = []
    for i in range(0,4):
        for j in range(i,(i+1)):
            hashVals_y=[]
            for hab in gen_e2LSH_family_y[i]:
                hashVal_y = (np.inner(hab[0], x_test2_pca[j]) + hab[1]) // r
                hashVals_y.append(hashVal_y)
            Y_h.append(hashVals_y)

    distance_m = []
    for z in range(samples_for_train):
        for x in range(samples_for_train):
            distance = euclideanDistance(X_h[z], X_h[x])
            distance_m.append(distance)
    theta = 0.8*max(distance_m)

    index_found_Z = []
    for y in range(76):
        min_distance = 1000000000
        for x in range(304):
            distance = euclideanDistance(Z_h[y], X_h[x])
            if distance < min_distance:
                min_distance = distance
                index = x
                if min_distance > theta:
                    min_distance = - distance
                    index = 0
        index_found_Z.append(index)
    

    index_found_Y = []
    for y in range(4):
        min_distance = 1000000000
        for x in range(samples_for_train):
            distance = euclideanDistance(Y_h[y], X_h[x])
            if distance < min_distance:
                min_distance = distance
                index = x
                if min_distance > theta:
                    min_distance = - distance
                    index = 800
        index_found_Y.append(index)

    
    data = np.array(index_found_Y)
    mask = np.unique(data)
    tmp = {}
    for v in mask:
        tmp[v] = np.sum(data == v)

    precision =precision+ (sum(labels_train[index_found_Z[0:76]] == labels_test[0:76])/76)

print("平均识别准确率", precision/repeat_times)
