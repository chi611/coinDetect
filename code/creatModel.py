import os
import csv
import cv2
import numpy as np

# 資料夾路徑
folder_path = '../coinData'

# CSV檔案路徑
csv_file_path = '../model/trainingModel.csv'

# 擷取所有影像的RGB像素值並寫入CSV檔案
with open(csv_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # 寫入CSV標題
    writer.writerow(['Image', 'RGB', 'Label'])

    # 讀取資料夾中的所有影像
    for folder in os.listdir(folder_path):

        for filename in os.listdir(folder_path + '/' + folder):
            print(filename)
            if filename.endswith('.jpg') or filename.endswith('.png'):  # 假設影像格式為.jpg或.png
                image_path = os.path.join(folder_path + '/' + folder, filename)

                # 讀取影像並提取RGB像素值
                image = cv2.imread(image_path)
                image = cv2.resize(image,(45,45))
                # 取得圖片中心點座標
                height, width = image.shape[:2]
                center = (width / 2, height / 2)

                # 定義旋轉角度和縮放比例
                scale = 1.0
                for angle in range(0,360,20):
                    # 構造旋轉矩陣
                    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

                    # 進行圖片旋轉，使用最近的邊界像素填充
                    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)

                    # 顯示旋轉後的圖片
                    cv2.imshow('image', rotated_image)
    
                    # 分割RGB通道並轉換為一維陣列
                    r_channel = list(rotated_image[:, :, 2].reshape(-1))  # 提取R通道
                    g_channel = list(rotated_image[:, :, 1].reshape(-1))  # 提取G通道
                    b_channel = list(rotated_image[:, :, 0].reshape(-1))  # 提取B通道
                    rgb = r_channel+g_channel+b_channel
                    # 將RGB像素值寫入CSV檔案
                    writer.writerow([filename] + [rgb] + [folder] )  # 寫入影像檔案名稱和像素值

print('CSV檔案寫入完成。')