import os

img_path = "word_10.png"


import paddle

# 正確的 GPU 偵測語法
# 修正：使用 paddle.device 而非 paddlex.device
has_gpu = paddle.device.is_compiled_with_cuda()
print(f"--- 偵測 GPU 編譯狀態: {has_gpu} ---")

if has_gpu:
    # 獲取當前 GPU 名稱
    current_dev = paddle.device.get_device()
    print(f"--- 當前使用設備: {current_dev} ---")
    run_device = "gpu:0"
else:
    print("--- 警告：未偵測到 GPU 或 CUDA 未正確安裝，將回退至 CPU 執行 ---")
    run_device = "cpu"


"""
https://www.paddleocr.ai/v3.3.2/version3.x/pipeline_usage/OCR.html#21
（1）通过 PaddleOCR() 实例化 OCR 产线对象，具体参数说明如下：
PP-OCRv5 必須使用 PaddlePaddle 3.x
而呼叫方法也變更，無法使用舊版 det/rec
PaddleOCR 呼叫的方式就是先偵測在辨識，一定會走2個模型，無法關閉
text_recognition_model_name="PP-OCRv5_server_det"
text_recognition_model_name="PP-OCRv5_server_rec"
"""
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    ocr_version="PP-OCRv5",
    use_angle_cls=True,
    lang="ch",
    device=run_device,
    # text_detection_model_name=None,  # <--- 並不會就關閉偵測功能，會使用預設 model_name(PP-OCRv5_server_det)
    text_recognition_model_name="PP-OCRv5_mobile_rec",
)

# === 執行 OCR ===
result = ocr.predict(img_path)
if not result:
    print("No result returned.")
    raise SystemExit

r = result[0]
print("回傳 keys:", r.keys())
# === 解析結果 ===
# 因為 r 是一個字典，我們直接從對應的 key 取值
texts = r.get("rec_texts", [])
scores = r.get("rec_scores", [])

settings = r.get("model_settings", {})
print("\n" + "=" * 50)
print("--- 完整 OCR 引擎設定資訊 ---")
# 根據文檔，新版 Pipeline 將模型實例封裝在底層
for res in result:
    res.print()

# 遍歷 result 物件，抓取內部的預測器資訊
for res in result:
    # 檢查該結果物件是否有紀錄當初使用的模型名稱
    if "model_settings" in res:
        print(f"使用的模型設定: {res['model_settings']}")

    # 這是最強招：直接檢查 ocr 執行實體
    if hasattr(ocr, "paddlex_pipeline"):
        # 抓取底層的 model_dir
        det_dir = getattr(
            ocr.paddlex_pipeline._pipeline.text_det_model, "model_name", "N/A"
        )
        rec_dir = getattr(
            ocr.paddlex_pipeline._pipeline.text_rec_model, "model_name", "N/A"
        )
        print(f"🔥 實體偵測模型路徑: {det_dir}")
        print(f"🔥 實體辨識模型路徑: {rec_dir}")


if texts:
    print("\n" + "=" * 30)
    for i in range(len(texts)):
        print(f"序號: {i+1}")
        print(f"辨識文字: {texts[i]}")
        print(f"信心評分: {scores[i]:.4f}")
    print("=" * 30)
else:
    print("未能辨識出任何文字。")


"""
https://www.paddleocr.ai/v3.3.2/version3.x/module_usage/text_recognition.html#_3
PaddlePaddle 3.x
paddleocr 新版的單獨呼叫模組
TextRecognition
"""
from paddleocr import TextRecognition

model = TextRecognition(model_name="PP-OCRv5_mobile_rec")
output = model.predict(input=img_path, batch_size=1)
print("\n" + "=" * 30)
print(f"--- PaddleOCR ---")
print("正在單獨加載 TextRecognition ...")

for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/res.json")


"""
PaddlePaddle 3.x
paddlex 新版單獨辨識的呼叫方式(這是AI幫我從 paddlex底層挖出來的)
TextRecPredictor
"""
from paddlex.inference.models.text_recognition.predictor import TextRecPredictor

# 1. 引擎版本確認 (應為 3.3.0)
print("\n" + "=" * 30)
print(f"--- PaddleX ---")
print(f"--- Paddle 引擎版本: {paddle.__version__} ---")

# 2. 定義模型與圖片路徑
v5_rec_model_dir = r"./models/PP-OCRv5_server_rec_infer"


try:
    print("正在單獨加載 PP-OCRv5 識別引擎 (Direct Predictor 模式)...")

    # 3. 直接初始化識別器 (不經過 create_pipeline)
    # 這個類別會直接讀取你的 inference.json 和 inference.pdiparams
    predictor = TextRecPredictor(model_dir=v5_rec_model_dir, device=run_device)

    # 4. 執行預測
    print(f"正在識別圖片: {img_path}")
    # predict 返回的是一個 generator，轉成 list 處理
    output = list(predictor.predict(img_path))

    # 5. 解析結果
    print("\n" + "=" * 30)
    for res in output:
        # PaddleX 3.x Predictor 的返回結構
        # 內含 'rec_text' 和 'rec_score'
        text = res.get("rec_text", "N/A")
        score = res.get("rec_score", 0.0)
        print(f"識別內容: {text}")
        print(f"信心評分: {score:.4f}")
        print("-" * 20)
        print("詳細輸出資料:", res)
    print("=" * 30)
except Exception as e:
    print(f"執行失敗，報錯訊息：{e}")
    import traceback

    traceback.print_exc()
