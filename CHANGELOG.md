# 更新日誌

專案的版本更新內容會被記錄在這個檔案

更新日誌的格式將會基於 [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
==============================================================================


## [1.2.0] - 2026-04-08
### Added
- PT2ONNX.py	將 .pt檔轉成 .onnx檔
- split_rec_gt_2_train_data.py	將 train_data 下的 rec_gt.txt 自動按照 0.75的比例分成 train_list.txt 和 val_list.txt
- YOLOv8OCR.py	將V3版本的YOLO-Pose 的.pt檔，改成ONNX

### Changed
- test_OCRv5.py	釐清新版 Paddle 3.x的呼叫方式和單獨模組 PaddleOCR 呼叫函式


## [1.1.0] - 2026-03-25
### Changed
- src_org	PaddleOCR-3.3.2原始碼移動到 src，使用呼叫程式碼的方式(和V3一樣)訓練


## [1.0.0] - 2026-03-19
### Added
- `專案結構文件樹->RUN&FILETREE.md`
- `#原始碼src: Fork from [PaddleOCR-3.3.2](https://github.com/PaddlePaddle/PaddleOCR/tree/v3.3.2)`


