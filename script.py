import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
with open('aqi_rf.pkl','rb') as file:
    rf_model = pickle.load(file)
with open('aqi_lr.pkl','rb') as file:
    lr_model = pickle.load(file)
with open('aqi_svr.pkl','rb') as file:
    svr_model = pickle.load(file)
with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)
with open('pca.pkl','rb') as file:
    pca = pickle.load(file)
filepath = r"C:\Users\Lenovo\Desktop\PROJECT_FINAL\Data_image\image\resize"
def sobeled(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    imgindnep3_sobel = cv2.magnitude(sobelx, sobely)
    imgindnep3_sobel = cv2.convertScaleAbs(imgindnep3_sobel)
    return imgindnep3_sobel
def selectroi(imggray, img, marg):
    imgindnep5 = img.astype(np.float32)/256.0
    imggray1 = imggray.astype(np.float32)/256.0
    minw = marg*marg
    roiy = 0
    roix = 0
    rangey = 0
    rangex = 0
    h, w = imgindnep5.shape
    while rangey+marg <= h and rangex+marg <= w:
        while rangex+marg <= w:
            minwc = np.sum(imgindnep5[rangey:rangey+marg,rangex:rangex+marg])
            if minwc < minw:
                minw = minwc
                roiy = rangey
                roix = rangex
            rangex += marg
        rangey += marg
        rangex = 0
    minmat = imggray1[roiy:roiy+marg,roix:roix+marg]
    minmat1d = minmat.reshape(-1)
    return roiy, roix, minmat1d
def selectroi2(imggray, img, marg, roiy, roix):
    imgindnep5 = img.astype(np.float32)/256.0
    imggray1 = imggray.astype(np.float32)/256.0
    minw = marg*marg
    roiy2 = 0
    if roiy == 0 and roix == 0:
        roix2 = marg
    else:
        roix2 = 0
    rangey = 0
    rangex = 0
    h, w = imgindnep5.shape
    while rangey+marg <= h and rangex+marg <= w:
        while rangex+marg <= w:
            minwc = np.sum(imgindnep5[rangey:rangey+marg,rangex:rangex+marg])
            if minwc < minw and roiy2 != roiy and roix2 != roix:
                minw = minwc
                roiy2 = rangey
                roix2 = rangex
            rangex += marg
        rangey += marg
        rangex = 0
    minmat = imggray1[roiy2:roiy2+marg,roix2:roix2+marg]
    minmat1d = minmat.reshape(-1)
    return roiy2, roix2, minmat1d

def encoder(filename, marg2):
    filename2 = cv2.imread(filename)
    if filename2 is None:
        raise ValueError(f"Cannot load image at path: {filename}")
    filename2gray = cv2.cvtColor(filename2, cv2.COLOR_BGR2GRAY)
    filename2red = filename2[:,:,2]
    filenamesobel = sobeled(filename2gray)
    arrboard = np.zeros((2, marg2*marg2))
    roiy, roix, roi1d = selectroi(filename2red, filenamesobel, marg2)
    roiy2, roix2, roi1d2 = selectroi2(filename2red, filenamesobel, marg2, roiy, roix)
    if roi1d.size == 0:
        roi1d = np.zeros(marg2*marg2)
    arrboard[0, :len(roi1d)] = roi1d[:marg2*marg2]
    arrboard[1, :len(roi1d)] = roi1d2[:marg2*marg2]
    return arrboard

def predictrf(c):
    c2 = encoder(c,512)
    c2_SS = scaler.transform(c2)
    c2_PCA = pca.transform(c2_SS)
    c2pred = rf_model.predict(c2_PCA)
    c2predexp = np.exp(c2pred)
    c2predavg = np.average(c2predexp)
    return c2predavg

def predictlr(c):
    c2 = encoder(c,512)
    c2_SS = scaler.transform(c2)
    c2_PCA = pca.transform(c2_SS)
    c2pred = lr_model.predict(c2_PCA)
    c2predexp = np.exp(c2pred)
    c2predavg = np.average(c2predexp)
    return c2predavg

def predictsvr(c):
    c2 = encoder(c,512)
    c2_SS = scaler.transform(c2)
    c2_PCA = pca.transform(c2_SS)
    c2pred = svr_model.predict(c2_PCA)
    c2predexp = np.exp(c2pred)
    c2predavg = np.average(c2predexp)
    return c2predavg

def pm25_to_aqi(pm):

    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]

    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= pm <= c_high:
            aqi = ((i_high - i_low) / (c_high - c_low)) * (pm - c_low) + i_low
            return round(aqi)

    return None

def aqi_level(aqi):
    if aqi <= 50:
        return "ดี", "good"
    elif aqi <= 100:
        return "ปานกลาง", "moderate"
    elif aqi <= 150:
        return "เริ่มมีผลกระทบ", "unhealthy1"
    elif aqi <= 200:
        return "มีผลต่อสุขภาพ", "unhealthy2"
    elif aqi <= 300:
        return "อันตราย", "danger"
    else:
        return "อันตรายมาก", "hazard"