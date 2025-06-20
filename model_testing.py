"""モデル、画像の入ったフォルダを指定してモデルの性能を検証する"""

import torch
import os
from PIL import Image

from common_utils import get_data_transforms, get_device
from model_factory import create_model

"""取得した画像に対し、作成したモデルを用いて OK/NGの分類を行う"""
def run_suiron(suiron_dir, model_file):
    if not os.path.exists(model_file):
        print(f"モデルファイルが見つかりません: {model_file}")
        return

    data_transforms = get_data_transforms()
    device = get_device()
    model = create_model()
    model.load_state_dict(torch.load(model_file))
    model = model.to(device)
    model.eval()

    class_names = ['NG', 'OK']

    for file_name in os.listdir(suiron_dir):
        file_path = os.path.join(suiron_dir, file_name)
        if not os.path.isfile(file_path):
            continue

        image = Image.open(file_path).convert("RGB")
        input_tensor = data_transforms(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            ng_prob = probabilities[0].item()
            ok_prob = probabilities[1].item()

            predicted_class = class_names[0] if ng_prob > ok_prob else class_names[1]
            ng_rate = ng_prob * 100 if predicted_class == "NG" else (1 - ok_prob) * 100

            print(f"ファイル名: {file_name}, クラス: {predicted_class}, NG率: {ng_rate:.2f}%")