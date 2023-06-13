import cv2
import numpy as np
import pickle
import time
# 圖片路徑
filename = "../image/input7.png"
# 模型檔路徑
model_filename = '../model/svm_model.pkl'
with open(model_filename, 'rb') as file:
    loaded_model = pickle.load(file)

# 預測硬幣圖像
def predict_coin(image):
    r_channel = list(image[:, :, 2].reshape(-1))  # 提取R通道
    g_channel = list(image[:, :, 1].reshape(-1))  # 提取G通道
    b_channel = list(image[:, :, 0].reshape(-1))  # 提取B通道
    rgb = r_channel+g_channel+b_channel    
    predicted_label = loaded_model.predict([rgb])

    return predicted_label

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
    #cv2.imshow('closing', closing)

    #==============邊緣檢測==============
    img_sobel = sobelEdgeDetection(img_gray)
    img_sobel = cv2.dilate(img_sobel,np.ones((3,3),dtype=np.uint8),iterations=1) 
    cnts,hierarchy=cv2.findContours(img_sobel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    cv2.drawContours(img_sobel, cnts, -1, 255, -1)
    #cv2.imshow('img_sobel', img_sobel)

    #========合併前兩者的硬幣抓取=========
    closing = cv2.add(closing, img_sobel)
    cnts,hierarchy=cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    cv2.drawContours(closing, cnts, -1, 255, -1)
    #cv2.imshow('coin', closing)

    # 確定背景區域 
    sure_bg = cv2.dilate(closing,np.ones((9,9),dtype=np.uint8),iterations=1) 
    #cv2.imshow('sure_bg', sure_bg)
    # 尋找前景區域 
    dist_transform = cv2.distanceTransform(closing,cv2.DIST_L2,5) 
    ret, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0) 
    #cv2.imshow('sure_fg', sure_fg)
    # 找到未知區域 
    sure_fg = np.uint8(sure_fg) 
    unknown = cv2.subtract(sure_bg,sure_fg)
    #cv2.imshow('unknown', unknown)
    # 類別標記 
    ret, markers = cv2.connectedComponents(sure_fg) 
    # 為所有的標記加1，保證背景是0而不是1 
    markers = markers+1 
    # 現在讓所有的未知區域為0 
    markers[unknown==255] = 0
    markers = cv2.watershed(img,markers) 

    result = np.zeros((720,1080),np.uint8)
    for i in range(2,ret + 1):
        result[markers == i] = 255
    result = cv2.erode(result,np.ones((5,5),dtype=np.uint8),iterations=1) 
    #cv2.imshow('result',result)
    return result


def main():
    img = cv2.imread(filename)
    img = cv2.resize(img,(1080,720))
    img_ori = img.copy()
    #cv2.imshow('img', img)
    result = coinDetect(img)
    #cv2.imshow('coinDetect', result)
    cnts,hierarchy=cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    coinAmount = [0, 0, 0, 0] #硬幣數量(list[0]: 50元數量、list[1]: 10元數量、list[2]: 5元數量、list[3]: 1元數量
    total = 0
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area <= 500:continue
        (x, y, w, h) = cv2.boundingRect(cnt)
        if y <20 or x <20 :continue

        cv2.circle(img, (int(x + w / 2), int(y + h / 2)), int((w + h) / 4), (0,0,255), 2)
        img_pic = img_ori[y-20:y+h+20,x-20:x+w+20]
        img_pic = cv2.resize(img_pic,(45,45))
        predicted_label = predict_coin(img_pic)

        if predicted_label == 11 or predicted_label == 12:
            cv2.putText(img, '1NT$', (x, y), cv2.FONT_HERSHEY_SIMPLEX,  1.3, (255, 0, 0), 3, cv2.LINE_AA)
            coinAmount[3] += 1
            total += 1
        elif predicted_label == 51 or predicted_label == 52:
            cv2.putText(img, '5NT$', (x, y), cv2.FONT_HERSHEY_SIMPLEX,  1.3, (255, 0, 0), 3, cv2.LINE_AA)        
            coinAmount[2] += 1
            total += 5
        elif predicted_label == 101 or predicted_label == 102 or predicted_label == 103:
            cv2.putText(img, '10NT$', (x, y), cv2.FONT_HERSHEY_SIMPLEX,  1.3, (255, 0, 0), 3, cv2.LINE_AA)  
            coinAmount[1] += 1
            total += 10
        elif predicted_label == 501 or predicted_label == 502:
            cv2.putText(img, '50NT$', (x, y), cv2.FONT_HERSHEY_SIMPLEX,  1.3, (255, 0, 0), 3, cv2.LINE_AA)            
            coinAmount[0] += 1
            total += 50

    print('總金額：',total)
    cv2.imshow('result', img)
start = time.time()
main()
end = time.time()
print('執行時間',end - start)
cv2.waitKey()
cv2.destroyAllWindows()