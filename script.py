import cv2
import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt
import sys
rel_files=[]
tif_files = []
def get_tif_files(folder_path):
    normalized_path = os.path.normpath(folder_path)
    for root, dirs, files in os.walk(normalized_path):
        for file in files:
            if file.lower().endswith('.tif'):
                tif_files.append(os.path.join(root, file))
                rel_files.append(file)
    return tif_files

folder_path = input('Enter folder path: ')
scale = int(input("Enter scale: "))
min_radius = int(input("Enter min_radius: "))
max_radius = int(input("Enter max_radius: "))
tif_files = get_tif_files(folder_path)
index=-1
D = []
for file_path in tif_files:
    index+=1
    print(file_path)
    # 读取图像
    image = cv2.imread(file_path)

    # 比例尺测量
    scale_image = image.copy()

    # 裁剪图像的左下角区域
    height, width, _ = scale_image.shape
    roi = scale_image  # 裁剪左下角区域，根据实际情况调整坐标和大小

    # 灰度化
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 二值化
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    #cv2.imwrite('binary.jpg', binary)
    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 通过轮廓筛选可能是比例尺的对象
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # 过滤掉太短或太窄的对象，根据实际情况调整阈值
        if w > 100 and h > 2:
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 0, 255), 2)
            print(x,y,w,h)
            scale_length_pixels = w  # 获取比例尺的像素长度
            break  # 假设只有一个比例尺，找到一个后就退出循环

    # 比例尺的真实长度（例如，1毫米）
    scale_length_mm = scale  # 假设比例尺的真实长度为10毫米

    # 计算像素到真实尺寸的转换比例
    pixel_to_mm_ratio = scale_length_mm / scale_length_pixels
    min_radius /= pixel_to_mm_ratio
    max_radius /= pixel_to_mm_ratio
    print(scale_length_pixels)

    filtered_image = image.copy()#方便最后画图并且缩放

    # 2. 复制图层
    layer = image.copy()

    # 3. 模糊当前图层
    blurred_layer = cv2.GaussianBlur(layer, (501, 501), 0)

    # 4. 调低亮度
    lowered_brightness = cv2.convertScaleAbs(blurred_layer, alpha=1, beta=-50)

    # 5. 图层混合（减去）
    blended_image = cv2.subtract(layer, lowered_brightness)

    # 6. 合并图
    merged_layer = cv2.add(layer, blended_image)

    # 7. 模糊当前图层
    blurred_merged_layer=cv2.convertScaleAbs(blended_image, alpha=-1., beta=90)
    blurred_merged_layer=cv2.convertScaleAbs(blurred_merged_layer, alpha=3., beta=0)


    # 8. 保存最终图片
    #output_path = 'blurred_image.jpg'
    #cv2.imwrite(output_path, blurred_merged_layer)


    gray_image = cv2.cvtColor(blurred_merged_layer, cv2.COLOR_BGR2GRAY)

    _, binary_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
    #cv2.imshow("binary_image", binary_image)
    # 2. 形态学操作 - 膨胀和腐蚀
    kernel = np.ones((5,5), np.uint8)
    dilated_image = cv2.dilate(binary_image, kernel, iterations=1)
    eroded_image = cv2.erode(dilated_image, kernel, iterations=2)

    # 3. 滤波器 - 高斯滤波器
    blurred_image = cv2.GaussianBlur(eroded_image, (3, 3), 0)
    # 4. 边缘检测和轮廓提取
    edges = cv2.Canny(blurred_image, threshold1=50, threshold2=100)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5. 绘制筛选后的轮廓



    # 计算直径数据
    diameters = []  # 存储圆的直径

    for contour in contours:
    # 计算轮廓的面积和周长
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # 如果面积和周长满足条件，绘制轮廓并计算直径
        if area>math.pi*min_radius**2 and area < math.pi*max_radius**2 and perimeter > math.pi*2*min_radius and perimeter < math.pi*2*max_radius:
            cv2.drawContours(filtered_image, [contour], -1, (0, 255, 0), 2)
            
            # 计算轮廓的最小外接圆
            (x, y), radius = cv2.minEnclosingCircle(contour)
            radius = (area/math.pi)**0.5
            # 过滤掉半径不满足条件的圆
            if radius >min_radius and radius<max_radius:
                cv2.circle(filtered_image, (int(x), int(y)), int(radius), (0, 0, 255), 2)
                diameter = radius * 2
                diameters.append(diameter)

    # 保存最终图片
    output_path = 'processed_'+rel_files[index]+'.jpg'
    cv2.imwrite(output_path, filtered_image)

    D.append(diameters)


# 创建数据
data = {'File': rel_files,
        'Diameter': D}
print(data)
dfs = []

# 遍历数据并将每个直径存储为一个 DataFrame 对象
for file, diameters in zip(data['File'], data['Diameter']):
    df = pd.DataFrame({'File': [file], 'Diameter': [diameters]})
    dfs.append(df)

# 合并所有 DataFrame
df = pd.concat(dfs, ignore_index=True)

# 导出为 Excel 文件
df.to_excel('output.xlsx', index=False)