import cv2
import numpy as np
# 邊緣檢測
def sobelEdgeDetection(f):
    grad_x = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize = 3)
    grad_y = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize = 3)
    magnitude = abs(grad_x) + abs(grad_y)
    g = np.uint8(np.clip(magnitude, 0, 255))
    ret, g = cv2.threshold(g, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return g

def coinDetect(img):
    #===============降噪================
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gauss = cv2.GaussianBlur(img_gray, (9, 9), 1)
    thresh, img_binary_G = cv2.threshold(img_gauss, 127, 255, cv2.THRESH_BINARY_INV)
    closing = cv2.morphologyEx(img_binary_G,cv2.MORPH_CLOSE,np.ones((3,3),dtype=np.uint8))
    cv2.imshow('closing', closing)

    #==============邊緣檢測==============
    img_sobel = sobelEdgeDetection(img_gray)
    img_sobel = cv2.dilate(img_sobel,np.ones((3,3),dtype=np.uint8),iterations=1) 
    cnts,hierarchy=cv2.findContours(img_sobel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    cv2.drawContours(img_sobel, cnts, -1, 255, -1)
    cv2.imshow('img_sobel', img_sobel)

    #========合併前兩者的硬幣抓取=========
    closing = cv2.add(closing, img_sobel)
    cnts,hierarchy=cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    cv2.drawContours(closing, cnts, -1, 255, -1)
    cv2.imshow('coin', closing)


    # 確定背景區域 
    sure_bg = cv2.dilate(closing,np.ones((9,9),dtype=np.uint8),iterations=1) 
    cv2.imshow('sure_bg', sure_bg)
    # 尋找前景區域 
    dist_transform = cv2.distanceTransform(closing,cv2.DIST_L2,5) 
    ret, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0) 
    cv2.imshow('sure_fg', sure_fg)
    # 找到未知區域 
    sure_fg = np.uint8(sure_fg) 
    unknown = cv2.subtract(sure_bg,sure_fg)
    cv2.imshow('unknown', unknown)
    # 類別標記 
    ret, markers = cv2.connectedComponents(sure_fg) 
    # 為所有的標記加1，保證背景是0而不是1 
    markers = markers+1 
    # 現在讓所有的未知區域為0 
    markers[unknown==255] = 0

    markers_temp = markers.copy()
    markers = np.uint8(markers) 
    cv2.imshow('markers', markers*30)
    markers = markers_temp
    markers = cv2.watershed(img,markers) 

    result = np.zeros((720,1080),np.uint8)
    for i in range(2,ret + 1):
        result[markers == i] = 255
  
    result = cv2.erode(result,np.ones((5,5),dtype=np.uint8),iterations=1) 
    cv2.imshow('result',result)
    return result


filename = "D:/school/embedded/image/input4.png"
img = cv2.imread(filename)
img = cv2.resize(img,(1080,720))
cv2.imshow('img', img)
result = coinDetect(img)

cnts,hierarchy=cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
for cnt in cnts:
    (x, y, w, h) = cv2.boundingRect(cnt)
    cv2.circle(img, (int(x + w / 2), int(y + h / 2)), int((w + h) / 4), (0,0,255), 2)
cv2.imshow('result', img)

cv2.waitKey()
cv2.destroyAllWindows()