import cv2
import numpy as np
import skimage.io as io
import os
import glob


########################sobel算子获得梯度特征####################
def sobel(img):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

    absX = abs(x)
    absY = abs(y)
    dst = cv2.addWeighted(absX,0.5,absY,0.5,0)
    return dst
############################查找真值图轮廓#########################
def contours(img,cnum):
    #img = cv2.GaussianBlur(img, (3, 3), 0)  # 用高斯平滑处理原图像降噪。若效果不好可调节高斯核大小
    contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)#寻找轮廓
    mask = np.zeros(img.shape, np.uint8)
    cv2.drawContours(mask, contours, -1, (255, 0, 0), cnum)  #最后一个参数为轮廓的层数,如果为2则一行有3个像素
    mask[0:3,:] = 0 ##############去除边界轮廓
    mask[:,0:3] = 0
    mask[img.shape[0]-3:img.shape[0],:] = 0
    mask[:,img.shape[1]-3:img.shape[1]] = 0
    return mask
#########################最大最小归一化##############################
def max_min(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
#########################均方差标准化##############################
def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

def means(data):
    _range = np.max(data) - np.min(data)
    return (data - np.mean(data, axis=0)) / _range