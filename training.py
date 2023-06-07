import cv2
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle

# 讀取CSV檔案
df = pd.read_csv('D:/school/embedded/data/trainingModel.csv')
X = list(map(eval, df.iloc[:, -2]))
y = df.iloc[:, -1].to_list()

# 訓練模型
svm_model = svm.SVC(kernel='linear')
svm_model.fit(X, y)

# 將模型保存到檔案
model_filename = 'D:/school/embedded/data/svm_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(svm_model, file)

# 預測新的硬幣圖像
def predict_coin(coin_image):
    # 加載模型檔案
    with open(model_filename, 'rb') as file:
        loaded_model = pickle.load(file)

    image = cv2.imread(coin_image)
    features = extract_features(image)

    # 進行預測
    predicted_label = loaded_model.predict([features])

    return predicted_label
def extract_features(image):
    image = cv2.resize(image,(45,45))
    # 分割RGB通道並轉換為一維陣列
    r_channel = list(image[:, :, 2].reshape(-1))  # 提取R通道
    g_channel = list(image[:, :, 1].reshape(-1))  # 提取G通道
    b_channel = list(image[:, :, 0].reshape(-1))  # 提取B通道
    rgb = r_channel+g_channel+b_channel
    return rgb

# 測試預測新的硬幣圖像
coin_image = 'D:/school/embedded/data/trainingModle/722227229208.png' #103
predicted_label = predict_coin(coin_image)
print("預測結果：", predicted_label)
coin_image = 'D:/school/embedded/data/trainingModle/412190147137.png' #501
predicted_label = predict_coin(coin_image)
print("預測結果：", predicted_label)
coin_image = 'D:/school/embedded/data/trainingModle/25552610699.png' #12
predicted_label = predict_coin(coin_image)
print("預測結果：", predicted_label)
coin_image = 'D:/school/embedded/data/trainingModle/363453111104.png' #51
predicted_label = predict_coin(coin_image)
print("預測結果：", predicted_label)
coin_image = 'D:/school/embedded/data/trainingModle/360335155127.png' #101
predicted_label = predict_coin(coin_image)
print("預測結果：", predicted_label)
coin_image = 'D:/school/embedded/data/trainingModle/257352138127.png' #103
predicted_label = predict_coin(coin_image)
print("預測結果：", predicted_label)
coin_image = 'D:/school/embedded/data/trainingModle/578474119101.png' #11
predicted_label = predict_coin(coin_image)
print("預測結果：", predicted_label)
