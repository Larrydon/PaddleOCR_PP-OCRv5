import os
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True' # 跳過連線檢查
import paddlex as pdx

# 在 v3 中，你可以直接使用 create_model 並指定模型名稱
# 這會自動從 PaddleX 的模型庫中尋找對應的架構
#model_name = "PP-OCRv5_server_rec"
model_name = "PP-HGNet_base"

try:
    # 1.建立模型實例
    model = pdx.create_model(model_name)
    print(f"成功建立模型: {model_name}")
    # 檢查模型是否具備 train 屬性
    if hasattr(model, 'train'):
        print(f"✅ 成功！{model_name} 現在支援訓練功能了。")
    else:
        print("❌ 依舊不支援訓練，請檢查 paddlex --install OCR 是否成功執行。")
    
    # 如果你想查看當前環境支援的所有 OCR 識別模型名稱
    # 在 v3 中通常透過搜尋工具或文檔查看，或者使用以下方式：
    # print(pdx.list_models()) 
    # 這函式不存在，會失敗，改到網站上查
    # https://paddlepaddle.github.io/PaddleX/latest/en/support_list/models_list.html?h=list+models#seal-text-detection-module


    # 2. 啟動訓練
    # 這裡會讀取你之前準備好的 TW_PP-OCRv5.yaml
    model.train(
        data_root="train_data",
        train_list="train_data/train_list.txt",
        val_list="train_data/val_list.txt",
        pretrain_weights="./models/PP-OCRv5_server_rec_pretrained.pdparams",
        save_dir="./output/TW_OCR_v5",
        epochs=100,
        batch_size=64
    )
    
except Exception as e:
    print(f"建立模型失敗: {e}")