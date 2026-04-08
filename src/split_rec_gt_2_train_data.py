import random
import json
import os

# --- 檔案路徑設定 ---
# 假設原始 PPOCRLabel 匯出的檔案是 Label.txt
# 但我們使用的是已經切割小圖的 rec_gt.txt
label_path = "./train_data/rec_gt.txt"

# 分割後輸出的檔案，用於 Rec 訓練
train_out = "./train_data/train_list.txt"
val_out = "./train_data/val_list.txt"

# Train / Val 的比例設定
train_ratio = 0.75

# 用於儲存轉換後的純文字行
converted_lines = []

# --- 讀取、轉換並格式化 ---
try:
    print(f"--- 正在讀取並轉換標註檔案: {label_path} ---")
    with open(label_path, "r", encoding="utf-8") as f:

        # 這是原本的 Label.txt 有 transcription JSON格式
        # for line in f:
        #     line = line.strip()
        #     if not line:
        #         continue

        #     # 1. 以 Tab 鍵分割檔名和 JSON 標註
        #     parts = line.split("\t", 1)
        #     if len(parts) != 2:
        #         print(f"警告：跳過格式不正確的行 (無Tab分隔符): {line}")
        #         continue

        #     image_filename_raw = parts[0]
        #     json_data_str = parts[1]

        #     # --- 彈性的路徑修正邏輯：僅保留檔名 ---
        #     # 使用 os.path.basename 從完整路徑中提取檔案名稱
        #     image_filename = os.path.basename(image_filename_raw)
        #     # ----------------------------------------

        #     try:
        #         # 2. 解析 JSON 陣列
        #         json_array = json.loads(json_data_str)

        #         # 3. 提取 transcription
        #         # PPOCRLabel 匯出的標註物件是一個陣列，我們處理其中的每一個物件
        #         transcriptions = []
        #         for item in json_array:
        #             if "transcription" in item:
        #                 transcriptions.append(item["transcription"])

        #         # 4. 針對 Rec 模型，我們只取第一個 transcription (假設每張圖只有一個車牌/文本行)
        #         if transcriptions:
        #             pure_text_label = transcriptions[0]
        #             # 5. 組合成新的簡潔格式：路徑 + Tab + 純文本
        #             new_line = f"images/{image_filename}\t{pure_text_label}\n"
        #             converted_lines.append(new_line)
        #         else:
        #             print(f"警告：{image_filename} 找不到有效的 transcription，跳過。")

        #     except json.JSONDecodeError:
        #         print(f"錯誤：JSON 解析失敗於 {image_filename}，跳過。")

        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            # 1. 以 Tab 鍵分割路徑和文字
            parts = line.split("\t")
            if len(parts) < 2:
                print(f"第 {line_num} 行警告：格式不正確 (無 Tab 分隔)，已跳過。")
                continue

            full_path = parts[0]
            label_text = parts[1]

            # 2. 提取純檔名 (例如: 202512171605560001_crop_0.jpg)
            image_filename = os.path.basename(full_path)

            # 3. 組合成你訓練需要的格式 (例如把 crop_img/ 改成 images/)
            # 如果你訓練時圖片就放在 images 資料夾，請維持此行
            new_line = f"images/{image_filename}\t{label_text}\n"
            converted_lines.append(new_line)

except FileNotFoundError:
    print(
        f"錯誤：找不到檔案 {label_path}。請確認您的 PPOCRLabel 匯出檔案是否已更名為 rec_gt.txt 並放置於 ./train_data/ 目錄下。"
    )
    exit()


# --- 分割與寫入檔案 ---
# 打亂資料（確保隨機性）
random.shuffle(converted_lines)

# 計算切割位置
train_count = int(len(converted_lines) * train_ratio)

train_lines = converted_lines[:train_count]
val_lines = converted_lines[train_count:]

# 寫回 train.txt (Rec 格式)
with open(train_out, "w", encoding="utf-8") as f:
    f.writelines(train_lines)

# 寫回 val.txt (Rec 格式)
with open(val_out, "w", encoding="utf-8") as f:
    f.writelines(val_lines)

print("-" * 30)
print("分割與轉換完成。")
print(f"轉換後總有效樣本數: {len(converted_lines)}")
print(f"訓練集 (Rec 格式) 樣本數: {len(train_lines)}")
print(f"驗證集 (Rec 格式) 樣本數: {len(val_lines)}")
print(f"訓練集檔案: {train_out}")
print(f"驗證集檔案: {val_out}")
