# -*- coding: utf-8 -*-
import numpy as np
import cv2

def canny(image):
    return cv2.Canny(image, 100, 200)


def anime_filter(image, K=20):
    # グレースケール変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    # ぼかしでノイズ低減
    edge = cv2.blur(gray, (3, 3))

    # Cannyアルゴリズムで輪郭抽出
    edge = cv2.Canny(edge, 50, 150, apertureSize=3) 

    # 輪郭画像をRGB色空間に変換
    edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

    # 画像の減色処理
    img = np.array(image/K, dtype=np.uint8)
    img = np.array(image*K, dtype=np.uint8)

    # 差分を返す
    return cv2.subtract(image, edge)

def make_contour_image(image):
    neiborhood24 = np.array([[1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1]],
                             np.uint8)
    # グレースケールで画像を読み込む.
    gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    #gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    #cv2.imwrite("gray.jpg", gray)

    # 白い部分を膨張させる.
    dilated = cv2.dilate(gray, neiborhood24, iterations=1)
    #cv2.imwrite("dilated.jpg", dilated)

    # 差をとる.
    diff = cv2.absdiff(dilated, gray)
    #cv2.imwrite("diff.jpg", diff)

    # 白黒反転
    image = 255 - diff
    #cv2.imwrite("./output.jpg", contour)
    return image

