# -*- coding: UTF-8 -*-     
from time import time
import logging
import matplotlib.pyplot as plt
import os
os.chdir('/Users/..../Documents/PycharmSource/libsvm-3.22/python')
from svmutil import *
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC


#打印程序进展信息
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')

#下载数据
lfw_people = fetch_lfw_people(min_faces_per_person=70,resize=0.4)

#数据集的实例个数
#introspect the images arrays to find the shapes(for poltting)
n_samples,h,w = lfw_people.images.shape

X = lfw_people.data

#特征的个数，列数
n_features = X.shape[1]

y = lfw_people.target
target_names = lfw_people.target_names
#有多少个人
n_classes = target_names.shape[0]

print('Total dataset size:')
print('n_sample：%d' %n_samples)
print('n_features:%d' %n_features)
print('n_classes:%d' %n_classes)

#通过调用函数，将数据集分成训练集和测试集两部分
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)

#降维
n_components = 150  #目标要降到的维数
t0 = time()
pca = RandomizedPCA(n_components=n_components,whiten=True).fit(X_train)
print('done in %0.3fs' %time()-t0)


print('Projecting the input data on the eigenfaces orthonormal vasis')
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.trandform(X_test)
print('done in %0.3fs' %(time()-t0))


#Train s SVM classification model

print('fitting the classifier to the training set')
t0 = time()
#C 惩罚系数，gamma：多少的特征点会被使用
param_grid = {'C':[1e3,5e3,1e4,5e4,1e5],'gamma':[0.0001,0.0005,0.001,0.01,0.1],} #5*6=30种组合，哪一种能得到最好的正确率

#
clf = GridSearchCV(SVC(kernel='rbf',class_weight='auto'),param_grid)

#建模
clf = clf.fit(X_train_pca,y_train)

print('done in %0.3fs' %(time()-t0))
print('best estimator found by grid search:')
print(clf.best_estimator_)


#测试集
print('predicting people names on the best set ')
t0 = time()
y_pred = clf.predict(X_test_pca)

print(classification_report(y_test,y_pred,target_names=target_names))
print(confusion_matrix(y_test,y_pred,labels=range(n_classes)))