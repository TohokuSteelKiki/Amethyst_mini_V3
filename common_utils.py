"""汎用的に使う関数のまとめ"""

import torch
from torchvision import transforms

"""GPUを使って演算するように指定"""
def get_device() -> str: 
    return "cpu"

"""ResNetモデルのフォーマットを最適化する"""
def get_data_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),  # ResNetの入力サイズにリサイズ
        transforms.ToTensor(),  # テンソルに変換
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正規化
    ])