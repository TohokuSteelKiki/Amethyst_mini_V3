import time
import cv2
import torch
from PIL import Image
import numpy as np

from common_utils import get_data_transforms, get_device
from model_factory import create_model
from image_process import maeshori_and_predict

# 使用するモデルファイルのパス（適切なパスに書き換えてください）
model_rasen_path = "Models/250523_rasen_00.pth"  # らせん疵用モデル
model_kurokawa_path = "Models/250523_rasen_00.pth"  # 黒皮残り用モデル

# 推論に必要な初期化
device = get_device()
data_transforms = get_data_transforms()
class_names = ['NG', 'OK']

# モデル構築とロード
model_rasen = create_model()
model_rasen.load_state_dict(torch.load(model_rasen_path, map_location='cpu'))  # CPU使用
model_rasen = model_rasen.to(device).eval()

model_kurokawa = create_model()
model_kurokawa.load_state_dict(torch.load(model_kurokawa_path, map_location='cpu'))  # CPU使用
model_kurokawa = model_kurokawa.to(device).eval()

# カメラの初期化
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("カメラを開けませんでした")
    exit()

while True:
    start_time = time.time()

    # カメラフレーム取得
    ret, frame = cap.read()
    if not ret:
        print("フレームを取得できませんでした")
        break

    # 必要ならリサイズ（224x448に合わせる）
    frame = cv2.resize(frame, (224, 448))

    # 映像を上下に分割
    frame_rasen = frame[:224, :]  # 上半分をらせん疵用に
    frame_kurokawa = frame[224:, :]  # 下半分を黒皮残り用に

    # らせん疵モデルで推論
    result_text_rasen, predicted_class_rasen, rasen_value, _ = maeshori_and_predict(
        frame_rasen, model_rasen, device, data_transforms, class_names)

    # 黒皮残りモデルで推論
    result_text_kurokawa, predicted_class_kurokawa, kurokawa_value, _ = maeshori_and_predict(
        frame_kurokawa, model_kurokawa, device, data_transforms, class_names)

    # 時間計測
    elapsed_time = time.time() - start_time
    fps = 1 / elapsed_time

    print(f"FPS: {fps:.2f}, 推論時間: {elapsed_time:.4f}秒")
    print(f"らせん疵: {predicted_class_rasen} ({rasen_value}%)")
    print(f"黒皮残り: {predicted_class_kurokawa} ({kurokawa_value}%)")

    # 30fpsに対して処理が追いついているかを確認
    if fps < 30:
        print("⚠ 推論処理が30fpsに追いついていません")
    else:
        print("✅ 推論処理は30fpsに追いついています")

    # 'q' で終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
