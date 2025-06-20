"""カメラの映像の上部・下部の明るさを参照して材料の有無を判定する"""

import cv2
import numpy as np

"""上部が明るい時にTrueを返す"""
def top_brightness(frame, brightness_threshold=15, area_ratio=0.1):
    height, width, _ = frame.shape
    end_y = int(height * area_ratio)  # 下部エリアの開始位置
    top_area = frame[:end_y, :]  # フレームの下部エリアを取得

    # グレースケール変換
    gray = cv2.cvtColor(top_area, cv2.COLOR_BGR2GRAY)

    # 平均輝度を計算
    top_avg_brightness = np.mean(gray)

    # デバッグ用：print(f"上部エリアの平均輝度: {top_avg_brightness:.0f}")
    return top_avg_brightness > brightness_threshold

"""下部が明るい時にTrueを返す"""
def bottom_brightness(frame, brightness_threshold=15, area_ratio=0.1):

    height, width, _ = frame.shape
    start_y = int(height * (1 - area_ratio))  # 下部エリアの開始位置
    bottom_area = frame[start_y:, :]  # フレームの下部エリアを取得

    # グレースケール変換
    gray = cv2.cvtColor(bottom_area, cv2.COLOR_BGR2GRAY)

    # 平均輝度を計算
    bottom_avg_brightness = np.mean(gray)

    # デバッグ用： print(f"下部エリアの平均輝度: {bottom_avg_brightness:.0f}")
    return bottom_avg_brightness > brightness_threshold