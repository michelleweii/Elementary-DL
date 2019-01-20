# coding=utf-8
# !/usr/bin/env python

import sys

path = r"H:\libsvm-3.22\libsvm-3.22\python"
sys.path.append(path)
from svmutil import *
from PIL import Image
import cv2
# from cv2 import cv
import numpy as np
from pylab import *
import glob
import os

'''
分割出图片中的数字，并改变大小，返回值为图片数组 ，如path = 'D:/pic/93.png'
返回的数据格式[[],[],[],.....]
'''


def picSplitResize(path):
    image = cv2.imread(path)
    ''' 获取一张图片四个角的灰度值'''
    h = image.shape[0]
    w = image.shape[1]
    print(w, h)

    # cv.GetSize(cv.fromarray(image))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    angleGray = [0] * 4
    angleGray[0] = gray[0][0]
    angleGray[1] = gray[0][w - 1]
    angleGray[2] = gray[h - 1][0]
    angleGray[3] = gray[h - 1][w - 1]
    num = 0
    for i in range(len(angleGray)):
        if angleGray[i] > 100:
            num += 1
    # print num
    ''' num大于3说明四个角是白色的，否则是黑色'''
    if num >= 3:
        # 把原来颜色反转后加强
        ret, bin = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    else:
        # 保持原来颜色不变，只是加强
        ret, bin = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    # cv2.imshow("bin",bin)
    # 膨胀后腐蚀
    dilated = cv2.dilate(bin, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    eroded = cv2.erode(dilated, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    cv2.imshow("median1", eroded)
    cv2.waitKey(0)
    # 腐蚀后膨胀
    # eroded = cv2.erode(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    # dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    # # 细化
    # median = cv2.medianBlur(dilated, 1)
    # median1 = cv2.medianBlur(dilated, 1)

    # 轮廓查找,查找前必须转换成黑底白字
    imageC, contours, heirs = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #print(what)

    i = 0
    pic = []
    dictPic = {}
    for tours in contours:
        rc = cv2.boundingRect(tours)
        # rc[0] 表示图像左上角的纵坐标，rc[1] 表示图像左上角的横坐标，rc[2] 表示图像的宽度，rc[3] 表示图像的高度，
        # cv2.rectangle(bin, (rc[0],rc[1]),(rc[0]+rc[2],rc[1]+rc[3]),(255,0,255))
        image1M = cv.fromarray(median)
        image1Ip = cv.GetImage(image1M)
        cv.SetImageROI(image1Ip, rc)
        imageCopy = cv.CreateImage((rc[2], rc[3]), cv2.CV_8U, 1)
        cv.Copy(image1Ip, imageCopy)
        cv.ResetImageROI(image1Ip)
        # print np.asarray(cv.GetMat(imageCopy))
        # 把图像左上角的纵坐标和图像的数组元素放到字典里
        dictPic[rc[0]] = np.asarray(cv.GetMat(imageCopy))
        pic.append(np.asarray(cv.GetMat(imageCopy)))
        # cv.ShowImage(str(i), imageCopy)
        # cv.Not(imageCopy, imageCopy)    #函数cvNot(const CvArr* src,CvArr* dst)会将src中的每一个元素的每一位取反，然后把结果赋给dst
        # cv.SaveImage(str(i)+ '.jpg',imageCopy)
        i = i + 1
    sortedNum = sorted(dictPic.keys())
    for i in range(len(sortedNum)):
        pic[i] = dictPic[sortedNum[i]]
    # cv2.waitKey(0)
    return pic


'''
#调整图片大小，先归一化图片，并把原图片放在中间,返回的数据格式 [[],[],.....]
'''


def resize(picArray, size):
    picNew = []
    for i in range(len(picArray)):
        imgPIL = Image.fromarray(picArray[i])
        h, w = imgPIL.size
        newH = w // 2 - h // 2  # 把图片放在中间
        imgEmpty = Image.new('L', (w, w), 0)  # 创建一张背景为黑色的图片
        imgEmpty.paste(imgPIL, (newH, 0))
        imgResize = imgEmpty.resize(size, Image.ANTIALIAS)
        imgResize0255 = imgResize.point(lambda x: 255 if x > 10 else 0)  # 0是黑，255是白    黑白加强
        imgResizeArray = array(imgResize0255).flatten().tolist()  # 转换为一维
        imgResizeArraySmaller = [float(x) / 255 for x in imgResizeArray]  # 把0-255转成0-1
        # print array(imgResize0255)
        picNew.append(imgResizeArraySmaller)
        # imgResize0255.show()
        imgResize0255.save(r'H:\StructureRecognition\pic\result' + str(i) + '.jpg')
    return picNew


'''
#获取训练数据,path为文件夹路径,suffix为图片后缀，图片名字的首字母必须是图片中的内容
'''


def traindata(path, suffix):
    train_images = []
    train_labels = []
    for files in glob.glob(path + '/*.' + suffix):
        filepath, filename = os.path.split(files)
        train_labels.append(int(filename[0:1]))
        pic = picSplitResize(filepath + '/' + filename)
        picNew = resize(pic, (20, 20))
        train_images.append(picNew[0])
        # im = Image.open(filepath + '/' + filename)
    return train_images, train_labels


'''
#创建LibSVM分类器,返回值为识别出的内容
'''


def predictPIC(train_images, train_labels, picdata):
    prob = svm_problem(train_labels, train_images)
    param = svm_parameter('-t 2')
    prob = svm_train(prob, param)
    labels = [0] * len(picdata)
    flag = svm_predict(labels, picdata, prob)
    print(flag[0])


pic = picSplitResize(r'H:/StructureRecognition/pic/sample/2.png')
# pic = picSplitResize('D:/pic/train and test/train/9-5.bmp')
# pic = picSplitResize('D:/pic/train and test/test1/7.bmp')
picdata = resize(pic, (20, 20))
# train_images,train_labels = traindata('D:/pic/train and test/train','bmp')
# result = predictPIC(train_images,train_labels,picdata)
