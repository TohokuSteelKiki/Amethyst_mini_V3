"""pyファイル分割前の残骸"""

from model_factory import train_model
from model_testing import run_suiron
from kizu_kenchi import display_realtime_suiron_with_separate_windows

###########################エントリーポイント###########################
if __name__ == "__main__":
    mode = input("モードを選択してください (train/suiron/realtime): ").strip().lower()

    if mode == "train":
        train_model()
    elif mode == "suiron":
        run_suiron()
    elif mode == "realtime":
        display_realtime_suiron_with_separate_windows()
    else:
        print("無効なモードが選択されました。train, suiron, または realtime を選んでください。")