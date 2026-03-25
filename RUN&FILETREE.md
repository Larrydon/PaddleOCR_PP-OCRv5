# 專案結構文件樹

- 專案結構文件樹
			(src）VSCode run Python，不改原始碼(PaddleOCR-3.3.2 原始碼)，改呼叫PaddleOCR程式碼來開發和訓練(和V3一樣)
				|----dict_taiwan_car.txt	台灣車牌字典，中英42字，和V3一樣
				|----draw_log.py	./output/TW_PP-OCRv5_Result/train.log 可視化 training_loss_plot.png
				|----PaddleX_API.py	測試用新的平台 PaddleX是否可以呼叫訓練V5成功? 都沒有成功才放棄這條路的，改回原本呼叫呼叫PaddleOCR程式碼的訓練
				|----test_OCRv5.py	用來測試新的呼叫API，可否正確使用 PP-OCRv5 來辨識成功
				|----TW_PP-OCRv5.yaml	V5是多頭式模型，因此不會有CPU的版本了(會很慢，有大量的CPU運算)，都使用GPU來訓練
 
- `.gitignore`
- `CHANGELOG.md`
- `README.md`
- `RUN&FILETREE.md`