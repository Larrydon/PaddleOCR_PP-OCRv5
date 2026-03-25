import matplotlib.pyplot as plt
import re

# 讀取你的 train.log 路徑
log_path = "./output/TW_PP-OCRv5_Result/train.log"
steps, losses, ctc_losses, nrtr_losses, accs = [], [], [], [], []

with open(log_path, "r", encoding="utf-8") as f:
    for line in f:
        if "global_step" in line:
            # 使用正則表達式抓取數據
            step = int(re.search(r"global_step: (\d+)", line).group(1))
            loss = float(re.search(r"loss: ([\d\.]+)", line).group(1))
            ctc = float(re.search(r"CTCLoss: ([\d\.]+)", line).group(1))
            nrtr = float(re.search(r"NRTRLoss: ([\d\.]+)", line).group(1))
            acc = float(re.search(r"acc: ([\d\.]+)", line).group(1))

            steps.append(step)
            losses.append(loss)
            ctc_losses.append(ctc)
            nrtr_losses.append(nrtr)
            accs.append(acc)

# 開始畫圖
plt.figure(figsize=(12, 5))

# 子圖 1: Loss 曲線
plt.subplot(1, 2, 1)
plt.plot(steps, losses, label="Total Loss", color="red")
plt.plot(steps, ctc_losses, label="CTC Loss", alpha=0.5)
plt.plot(steps, nrtr_losses, label="NRTR Loss", alpha=0.5)
plt.title("Training Loss (PP-OCRv5)")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()

# 子圖 2: Accuracy 曲線
plt.subplot(1, 2, 2)
plt.plot(steps, accs, label="Accuracy", color="green")
plt.title("Training Accuracy")
plt.xlabel("Steps")
plt.ylabel("Acc")
plt.legend()

plt.tight_layout()
# 3. 🚩 關鍵修改：儲存成檔案而不是顯示視窗
output_filename = "./output/TW_PP-OCRv5_Result/training_loss_plot.png"
plt.savefig(output_filename, dpi=300) # 設定 300 DPI 讓圖片更清晰
print(f"✅ 圖片已成功儲存至: {output_filename}")

# 不要呼叫 plt.show()，避免遠端報錯
# plt.show()
