# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import math

def allMax(img):
    maxForRow = []
    W,H = img.shape
    for i in range(W):
        maxForRow.append(max(img[i,:]))
    return (max(maxForRow))

def allMin(img):
    minForRow = []
    W,H = img.shape
    for i in range(W):
        minForRow.append(min(img[i,:]))
    return (min(minForRow))  

img = plt.imread('tools.png')


sigma1 = sigma2 = 1
sum = 0

gaussian = np.zeros([5, 5])
for i in range(5):
    for j in range(5):
        gaussian[i,j] = math.exp(-1/2 * (np.square(i-3)/np.square(sigma1)           #生成二维高斯分布矩阵
                        + (np.square(j-3)/np.square(sigma2)))) / (2*math.pi*sigma1*sigma2)
        sum = sum + gaussian[i, j]
        
gaussian = gaussian/sum

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])



# step1.高斯滤波
gray = rgb2gray(img)

W, H = gray.shape

newGray = np.zeros([W-5, H-5])
for i in range(W-5):
    for j in range(H-5):
        newGray[i,j] = np.sum(gray[i:i+5,j:j+5]*gaussian)   # 与高斯矩阵卷积实现滤波 
#print(newGray)

     
        
# step2.增强 通过求梯度幅值
W1, H1 = newGray.shape
dx = np.zeros([W1-1, H1-1])
dy = np.zeros([W1-1, H1-1])
d = np.zeros([W1-1, H1-1])
for i in range(W1-1):
    for j in range(H1-1):   
        dx[i,j] = newGray[i, j+1] - newGray[i, j]
        dy[i,j] = newGray[i+1, j] - newGray[i, j]        
        d[i, j] = np.sqrt(np.square(dx[i,j]) + np.square(dy[i,j]))   # 图像梯度幅值作为图像强度值
#print(d)        
         

      
        
# setp3.非极大值抑制 NMS
W2, H2 = d.shape
NMS = np.copy(d)
NMS[0,:] = NMS[W2-1,:] = NMS[:,0] = NMS[:, H2-1] = 0
for i in range(1, W2-1):
    for j in range(1, H2-1):
        
        if d[i, j] == 0:
            NMS[i, j] = 0
        else:
            gradX = dx[i, j]
            gradY = dy[i, j]
            gradTemp = d[i, j]
            
            # 如果Y方向幅度值较大
            if np.abs(gradY) > np.abs(gradX):
                weight = np.abs(gradX) / np.abs(gradY)
                grad2 = d[i-1, j]
                grad4 = d[i+1, j]
                # 如果x,y方向梯度符号相同
                if gradX * gradY > 0:
                    grad1 = d[i-1, j-1]
                    grad3 = d[i+1, j+1]
                # 如果x,y方向梯度符号相反
                else:
                    grad1 = d[i-1, j+1]
                    grad3 = d[i+1, j-1]
                    
            # 如果X方向幅度值较大
            else:
                weight = np.abs(gradY) / np.abs(gradX)
                grad2 = d[i, j-1]
                grad4 = d[i, j+1]
                # 如果x,y方向梯度符号相同
                if gradX * gradY > 0:
                    grad1 = d[i+1, j-1]
                    grad3 = d[i-1, j+1]
                # 如果x,y方向梯度符号相反
                else:
                    grad1 = d[i-1, j-1]
                    grad3 = d[i+1, j+1]
        
            gradTemp1 = weight * grad1 + (1-weight) * grad2
            gradTemp2 = weight * grad3 + (1-weight) * grad4
            if gradTemp >= gradTemp1 and gradTemp >= gradTemp2:
                NMS[i, j] = gradTemp
            else:
                NMS[i, j] = 0
#print(NMS)   

#用Ostu法选取阈值

#统计灰度频率
#gray = rgb2gray(img)
#W, H = gray.shape
#python里的灰度图灰度值是0-1的，要用0-255的灰度等级需要先归一化再扩倍
grayMax = allMax(d)
grayMin = allMin(d)
normGray = np.zeros([W2,H2])
for i in range(W2):
    for j in range(H2):
        normGray[i,j] = (d[i,j]-grayMin)/(grayMax-grayMin) * 255
#print(normGray)

grayScale = np.zeros(256)#每个灰度等级的像素点个数
grayScaleStatis = set()#便于统计的set
for i in range(W2):
    for j in range(H2):
        if int(normGray[i,j]) in grayScaleStatis :
            grayScale[int(normGray[i,j])] = grayScale[int(normGray[i,j])] + 1
        else:
            grayScaleStatis.add(int(normGray[i,j]))
#print(grayScale)    
#print(grayScaleStatis)

proGaryScale = np.zeros(256)#每个灰度等级的像素点的频率
numPixel = W2*H2      
for i in range(256):
    proGaryScale[i] = grayScale[i]/numPixel
#print(proGaryScale)


#灰度等级均值
mu = 0
for i in range(256):
    mu = mu + i * proGaryScale[i]
#寻找最佳划分阈值k
sigma01 = 0
kstar = 0
for k in range(256):
    omega0 = 0
    muk = 0
    for i in range(k+1):
        omega0 = omega0 + proGaryScale[i]
        muk = muk + i * proGaryScale[i]
    omega1 = 1 - omega0
    mu0 = muk / omega0
    mu1 = (mu - muk) / omega1 
    if (omega0 * omega1 * (mu0 - mu1) * (mu0 - mu1)) > sigma01 :
        sigma01 = omega0 * omega1 * (mu0 - mu1) * (mu0 - mu1)
        kstar = k
        mu0star = mu0
print("kstar: %f" % (kstar*(grayMax-grayMin)/255+grayMin))
sigma01Ori = math.sqrt(sigma01)*(grayMax-grayMin)/255+grayMin
muOri = mu*(grayMax-grayMin)/255+grayMin
#print(mu0star*(grayMax-grayMin)/255+grayMin)
print("sigma01Ori: %f" % sigma01Ori)
print("muOri: %f" % muOri)


# step4. 双阈值算法检测、连接边缘
W3, H3 = NMS.shape
DT = np.zeros([W3, H3])               
# 定义高低阈值
TL = kstar*(grayMax-grayMin)/255+grayMin - sigma01Ori 
TH = kstar*(grayMax-grayMin)/255+grayMin + sigma01Ori
for i in range(1, W3-1):
    for j in range(1, H3-1):
        if (NMS[i, j] < TL):
            DT[i, j] = 0
        elif (NMS[i, j] > TH):
            DT[i, j] = 1
        elif ((NMS[i-1, j-1:j+1] < TH).any() or (NMS[i+1, j-1:j+1]).any() 
              or (NMS[i, [j-1, j+1]] < TH).any()):
            DT[i, j] = 1

print("TL: %f" % TL)
print("TH: %f" % TH)

plt.imshow(DT, cmap = "gray")
#plt.savefig('tools_manual.png')
plt.show()

