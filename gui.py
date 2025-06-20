"""GUIの全て。いずれ読みやすいように分割する"""

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

# from model_factory import train_model

from kizu_kenchi import display_realtime_suiron_with_separate_windows

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Amethyst 汎用疵検知システム)")
        self.root.geometry("600x600")

        # メインレイアウトを設定
        self.setup_sidebar()
        self.setup_main_frame()

    def setup_sidebar(self):
        self.sidebar = ttk.Frame(self.root, width=200)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        # サイドバーとメインフレームの区切り線
        separator = ttk.Separator(self.root, orient="vertical")
        separator.pack(side=tk.LEFT, fill=tk.Y)

        # スタイルの定義
        style = ttk.Style()
        style.configure("SidebarButton.TButton", font=("Helvetica", 10))  # 全ボタン共通のフォントサイズ
        style.configure("RealtimeButton.TButton", background="#9E76B4", foreground="black", font=("Helvetica", 10))  # ゴールド背景、黒文字

        # モード選択ボタンを追加
        ttk.Label(self.sidebar, text="モードを選択", font=("Helvetica", 12)).pack(pady=10)

        self.realtime_button = ttk.Button(self.sidebar, text="疵検知モード (CM用)", command=self.realtime_mode, style="RealtimeButton.TButton")
        self.realtime_button.pack(fill=tk.X, padx=10, pady=5, ipadx=10, ipady=5)  # 内側余白を追加


    def setup_main_frame(self):
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        self.default_message = ttk.Label(
            self.main_frame, text="サイドバーからモードを選択してください。", font=("Helvetica", 12)
        )
        self.default_message.pack(expand=True)

    def clear_main_frame(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()

    def realtime_mode(self):
        self.clear_main_frame()
        ttk.Label(self.main_frame, text="疵検知モード", font=("Helvetica", 14)).pack(pady=10)

        # モデルファイル選択
        ttk.Label(self.main_frame, text="hefモデルファイル:").pack(anchor=tk.W, padx=10, pady=5)
        model_file_frame = ttk.Frame(self.main_frame)
        model_file_frame.pack(padx=10, pady=5, fill=tk.X)
        model_file_entry = ttk.Entry(model_file_frame, width=50)
        model_file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        browse_button = ttk.Button(model_file_frame, text="参照", command=lambda: self.select_file(model_file_entry))
        browse_button.pack(side=tk.RIGHT, ipadx=0, ipady=0, padx=(0, 20))  # 右揃えから左に20px移動

        # 自動保存設定
        ttk.Label(self.main_frame, text="自動保存:推論結果が下限%～上限%の間の時に保存します").pack(anchor=tk.W, padx=10, pady=5)
        auto_save_var = tk.BooleanVar(value=False)
        auto_save_checkbox = ttk.Checkbutton(self.main_frame, text="有効", variable=auto_save_var)
        auto_save_checkbox.pack(anchor=tk.W, padx=10)

        # 自動保存閾値設定
        ttk.Label(self.main_frame, text="自動保存の閾値 NG率(下限%, 上限%):").pack(anchor=tk.W, padx=10, pady=5)
        threshold_frame = ttk.Frame(self.main_frame)
        threshold_frame.pack(padx=10, pady=5, fill=tk.X)
        ng_threshold_entry = ttk.Entry(threshold_frame, width=10)
        ng_threshold_entry.pack(side=tk.LEFT, padx=5)
        ng_threshold_entry.insert(0, "50")
        ok_threshold_entry = ttk.Entry(threshold_frame, width=10)
        ok_threshold_entry.pack(side=tk.LEFT, padx=5)
        ok_threshold_entry.insert(0, "100")

        # クールタイム設定
        ttk.Label(self.main_frame, text="自動保存のクールタイム (秒):").pack(anchor=tk.W, padx=10, pady=5)
        cool_time_entry = ttk.Entry(self.main_frame, width=10)
        cool_time_entry.pack(anchor=tk.W, padx=10)
        cool_time_entry.insert(0, "0.5")

        # NG率差分閾値設定
        ttk.Label(self.main_frame, text="NG率差分閾値 (%):").pack(anchor=tk.W, padx=10, pady=5)
        ng_diff_threshold_entry = ttk.Entry(self.main_frame, width=10)
        ng_diff_threshold_entry.pack(anchor=tk.W, padx=10)
        ng_diff_threshold_entry.insert(0, "2")

        #brightness_thresholdの設定
        ttk.Label(self.main_frame, text="平均輝度 閾値:").pack(anchor=tk.W, padx=10, pady=5)
        brightness_threshold_entry = ttk.Entry(self.main_frame, width=10)
        brightness_threshold_entry.pack(anchor=tk.W, padx=10)
        brightness_threshold_entry.insert(0, "10")                

        #area_ratioの設定
        ttk.Label(self.main_frame, text="参照エリア(0～1.0)):").pack(anchor=tk.W, padx=10, pady=5)
        area_ratio_entry = ttk.Entry(self.main_frame, width=10)
        area_ratio_entry.pack(anchor=tk.W, padx=10)
        area_ratio_entry.insert(0, "0.05")

        # リアルタイム開始ボタンの追加（サイズとスタイルを変更）
        realtime_button_frame = tk.Frame(self.main_frame, bg="#8630B6")  # フレーム背景色を設定
        realtime_button_frame.place(relx=1, rely=1, anchor="se", x=-20, y=-20)  # 右下に固定

        realtime_button = tk.Button(
            realtime_button_frame,
            text="Start",
            bg="#9E76B4",  # 背景色を設定
            fg="#FFFFFF",  # 必要なら文字色を設定
            font=("Helvetica", 20),  # フォントサイズを設定
            command=lambda: self.start_realtime(
                model_file_entry.get().strip(), auto_save_var.get(),
                ng_threshold_entry.get(), ok_threshold_entry.get(),
                cool_time_entry.get(), ng_diff_threshold_entry.get(),
                brightness_threshold_entry.get(), area_ratio_entry.get()
            )
        )
        realtime_button.pack(ipadx=40, ipady=2)  # 内部余白を指定してボタンを大きく
  
    def select_directory(self, entry_widget):
        directory = filedialog.askdirectory()
        if directory:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, directory)

    def select_file(self, entry_widget):
        file_path = filedialog.askopenfilename(filetypes=[("Model Files", "*.hef")])
        if file_path:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, file_path)

    # def start_training(self, train_dir, model_name):
    #     if not train_dir or not model_name:
    #         print("学習ディレクトリとモデル名を入力してください。")
    #         return

    #     # モデル名に拡張子を追加
    #     if not model_name.endswith(".pth"):
    #         model_name += ".pth"

    #     train_model(train_dir, model_name)

    def start_realtime(self,
                        model_file, auto_save, 
                        ng_threshold, ok_threshold, 
                        cool_time, ng_diff_threshold, 
                        brightness_threshold, area_ratio
                        ):
        if not model_file:
            print("リアルタイム推論のモデルファイルを指定してください。")
            return

        try:
            ng_threshold = int(ng_threshold)
            ok_threshold = int(ok_threshold)
            cool_time = float(cool_time)
            ng_diff_threshold = float(ng_diff_threshold)
            brightness_threshold = int(brightness_threshold)
            area_ratio = float(area_ratio)
        except ValueError:
            print("入力された値が正しくありません。")
            return

        display_realtime_suiron_with_separate_windows(
            model_file, auto_save, 
            (ng_threshold, ok_threshold), 
            cool_time, ng_diff_threshold,
            brightness_threshold, area_ratio
            )
        
if __name__ == "__main__":
    import sys
    print(f"Pythonバージョン：{sys.version}")   # Pythonのバージョンを表示

    root = tk.Tk()
    app = App(root)
    root.geometry("600x600")
    root.mainloop()
