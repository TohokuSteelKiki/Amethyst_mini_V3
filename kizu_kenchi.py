"""疵検知: リアルタイム推論モジュール（mini版）"""
import cv2
import numpy as np
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from check_brightness import top_brightness, bottom_brightness
from image_process import save_photo
# Hailo ランナーと前処理関数（crop 引数付き）をインポート
from hailo_runner import HailoRunner, preprocess_rgb224


# -------------------- GStreamer パイプラインを動的生成 --------------------
def build_pipeline(width: int = 224,
                   height: int = 448,
                   cam_id: str = "SENTECH-142124706912-24G6912") -> str:
    """
    * Aravis (USB3-Vision) が存在すれば aravissrc を使用
    * それ以外は UVC (v4l2src) に自動フォールバック
    どちらの場合も videoscale + videoconvert で 224×448 BGR に揃える
    """
    try:
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        Gst.init(None)                                   # ←★ 追加 ★
        has_aravis = Gst.ElementFactory.find("aravissrc") is not None
    except Exception:
        has_aravis = False

    if has_aravis:
        return (
            f"aravissrc camera-name={cam_id} ! "
            "videoconvert ! videoscale ! "
            f"video/x-raw,format=BGR,width={width},height={height} ! "
            "appsink drop=true max-buffers=1"
        )
    else:
        # UVC カメラ想定
        return (
            "v4l2src device=/dev/video0 ! "
            "videoconvert ! videoscale ! "
            f"video/x-raw,width={width},height={height} ! "
            "appsink drop=true max-buffers=1"
        )
# -----------------------------------------------------------------------


def draw_text_with_japanese(image, text, position, font_path, font_size, color):
    """OpenCV画像上に日本語テキストを描画"""
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, fill=color, font=font)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def create_gauge_bar(percent, length=20):
    """ビビリ率に応じたゲージバー文字列を生成"""
    filled = int(length * percent / 100)
    bar = "|" * filled + "-" * (length - filled)
    return f"[{bar}]\n0%{' ' * (length//2 - 2)}50%{' ' * (length//2 - 3)}100%"


def get_gauge_color(percent):
    """ビビリ率に応じたバーの色を返す"""
    if percent <= 50:
        return (0, 255, 0)
    elif percent <= 80:
        return (255, 150, 0)
    else:
        return (255, 0, 0)


def display_realtime_suiron_with_separate_windows(
    model_rasen_path: str,
    model_kurokawa_path: str,
    *,
    auto_save: bool,
    auto_save_threshold: tuple[int, int],
    cool_time_seconds: float,
    ng_rate_diff_threshold: float,
    brightness_threshold: int,
    area_ratio: float,
    use_ng_diff: bool
):
    """
    2つのHEFモデルを同一デバイスでロード・推論し、
    OpenCVウィンドウにライブ表示する。
    224×448 のフレームを上下 224px ずつに分けて
    ・上段 → らせん疵モデル
    ・下段 → 黒皮残りモデル
    へ入力する。
    """
    # モデルランナーを 1台のデバイス上にそれぞれ生成
    runner_rasen = HailoRunner(model_rasen_path)
    runner_kurokawa = HailoRunner(model_kurokawa_path)

    # カメラ取得パイプラインを自動生成（U3V/UVC 両対応）
    pipeline = build_pipeline(width=224, height=448)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("カメラが開けません")
        return

    last_save_time = datetime.now()
    last_ng_rate = None
    cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Preview", 448, 448)
    cv2.namedWindow("Information", cv2.WINDOW_NORMAL)
    font_path = "fonts/PixelMplus10-Regular.ttf"
    font_size = 24

    while True:

        ret, frame = cap.read()  # frame.shape == (448, 224, 3) BGR
        if not ret:
            print("フレーム取得失敗")
            break
        tensor_top = preprocess_rgb224(frame, crop='top')
        print("tensor_top.shape:", tensor_top.shape, "size(bytes):", tensor_top.nbytes)
        tensor_bot = preprocess_rgb224(frame, crop='bottom')
        print("tensor_bot.shape:", tensor_bot.shape, "size(bytes):", tensor_bot.nbytes)
        print("frame.shape:",frame.shape)
        # プレビュー表示
        cv2.imshow("Preview", frame)

        # 明るさ検査
        if not top_brightness(frame, brightness_threshold, area_ratio) or \
           not bottom_brightness(frame, brightness_threshold, area_ratio):
            detection_text = "材料: ナシ"
            color_det = (0, 128, 0)
        else:
            detection_text = "材料: アリ"
            color_det = (0, 255, 0)

        # ---------- 前処理＆推論 : 上下 2 分割 ----------
        tensor_top = preprocess_rgb224(frame, crop='top')      # 上段 224×224
        tensor_bot = preprocess_rgb224(frame, crop='bottom')   # 下段 224×224

        # らせん疵モデル（上段）
        probs_r = runner_rasen.infer(tensor_top)
        ng_r, ok_r = probs_r / probs_r.sum()
        pred_r = "NG" if ng_r > ok_r else "OK"
        val_r = int(ng_r * 100)

        # 黒皮残りモデル（下段）
        probs_k = runner_kurokawa.infer(tensor_bot)
        ng_k, ok_k = probs_k / probs_k.sum()
        pred_k = "NG" if ng_k > ok_k else "OK"
        val_k = int(ng_k * 100)

        # -------- 描 画 --------
        result_text = f"【らせん】{pred_r} {val_r}%  【黒皮】{pred_k} {val_k}%"
        result_img = np.zeros((180, 600, 3), dtype=np.uint8)

        # メインテキスト
        result_img = draw_text_with_japanese(
            result_img, result_text, (10, 10),
            font_path, font_size, (255, 255, 255)
        )
        # ゲージ：らせん
        result_img = draw_text_with_japanese(
            result_img, create_gauge_bar(val_r, length=20),
            (10, 50), font_path, font_size, get_gauge_color(val_r)
        )
        # ゲージ：黒皮
        result_img = draw_text_with_japanese(
            result_img, create_gauge_bar(val_k, length=20),
            (320, 50), font_path, font_size, get_gauge_color(val_k)
        )
        # 検出マテリアル
        result_img = draw_text_with_japanese(
            result_img, detection_text, (10, 130),
            font_path, font_size, color_det
        )
        cv2.imshow("Information", result_img)

        key = cv2.waitKey(1) & 0xFF

        # 手動保存
        if key == ord('s') and auto_save:
            save_photo(frame.copy(), result_text, max(val_r, val_k), "images/Photo/manual")

        # 自動保存
        if auto_save and detection_text == "材料: アリ":
            current = datetime.now()
            elapsed = (current - last_save_time).total_seconds()

            # 差分チェック
            ng_diff = abs(max(val_r, val_k) - last_ng_rate) if last_ng_rate is not None else float('inf')
            if not use_ng_diff:
                ng_diff = float('inf')

            if elapsed >= cool_time_seconds and ng_diff > ng_rate_diff_threshold and \
               auto_save_threshold[0] <= max(val_r, val_k) <= auto_save_threshold[1]:
                save_photo(frame.copy(), result_text, max(val_r, val_k), "images/photo/auto")
                last_save_time = current
                last_ng_rate = max(val_r, val_k)

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
