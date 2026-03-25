# Fork from [PaddleOCR-3.3.2](https://github.com/PaddlePaddle/PaddleOCR/tree/v3.3.2)

### PaddleOCRv5/PaddleOCR-VL/PaddleOCR-VL-1.5<br>
https://github.com/PaddlePaddle/PaddleOCR/releases/tag/v3.3.2<br>
(docs\Release-v3-3-2-PaddlePaddle-PaddleOCR-github.com.mhtml)
<br>
<br>
將使用 PP-OCRv5 版本中的 [PP-OCRv5_server_rec](https://www.paddleocr.ai/v3.3.2/version3.x/module_usage/text_recognition.html#_3) 模型來開發車牌辨識<br>
PP-OCRv5_server_rec	PP-OCRv5_rec 是新一代文本识别模型。该模型致力于以单一模型高效、精准地支持简体中文、繁体中文、英文、日文四种主要语言，以及手写、竖版、拼音、生僻字等复杂文本场景的识别。在保持识别效果的同时，兼顾推理速度和模型鲁棒性，为各种场景下的文档理解提供高效、精准的技术支撑。	configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml	81MB<br>
[推理模型](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0//PP-OCRv5_server_rec_infer.tar)
[训练模型](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_rec_pretrained.pdparams)
<br>
<br>
參考來源:
模型庫 https://www.paddleocr.ai/v3.3.2/version3.x/module_usage/text_recognition.html<br>
(docs\PaddleOCR-3.3.2_文本識別模塊.mhtml)
<br>
<br>
<br>
<br>
## 環境設定
Python:3.9.25

### 1. 完全清理
> pip uninstall paddlepaddle paddleocr paddlepaddle-gpu paddlex paddlehub ppocronnx -y
> pip cache purge

<br>
<br>

### 2. 安裝符合 PaddleOCR-v3.3.2 要求的 PaddlePaddle (新引擎 PP-OCRv5/PaddleOCR-VL/PaddleOCR-VL-1.5 需 >= 3.0.0)
官網-快速開始<br>
https://www.paddleocr.ai/v3.3.2/version3.x/installation.html<br>

### 2.Step 1: 安裝核心引擎 (PaddlePaddle/PaddlePaddle-GPU)<br>
##### CPU 版本 3.2.2
##### 使用官方指定的 CPU 鏡像源安裝 3.2.2 正式版
>	pip install paddlepaddle==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

##### GPU 版本 3.0.0(請依 CUDA 版本調整)
##### 安裝支援 CUDA 11.8 的 PaddleOCR-v3.0.0 rc1 版本(根據您的 CUDA 調整，假設您的 CUDA 是 11.x)
>	pip install paddlepaddle-gpu==3.0.0rc1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

=>	我們的YOLO伺服器CPU太舊了(Intel(R) Xeon(R) CPU E5-1620)，無法支援到AVX2<br>
	檢查CPU是否支援AVX和AVX2:<br>
	cat /proc/cpuinfo | grep -i avx<br>
	cat /proc/cpuinfo | grep -E "avx2|fma"<br>
	這兩個都要有顯示才會是有支援<br>

<br>
安裝完後，建議輸入<br>
測試 AVX/AVX2 指令集

> python -c "import paddle; paddle.utils.run_check()"

測試，看到 PaddlePaddle is installed successfully! 再往下走。
>	
    (py309OCRv5) yolo@yolo-ESC700-G2:~/PaddleOCRv5_project$ python -c "import paddle; paddle.utils.run_check()"
	/home/yolo/anaconda3/envs/py309OCRv5/lib/python3.9/site-packages/paddle/utils/cpp_extension/extension_utils.py:711: UserWarning: No ccache found. Please be aware that recompiling all source files may be required. You can download and install ccache from: https://github.com/ccache/ccache/blob/master/doc/INSTALL.md
	  warnings.warn(warning_message)
	Running verify PaddlePaddle program ... 
	I0318 18:01:18.001847 1885944 pir_interpreter.cc:1508] New Executor is Running ...
	W0318 18:01:18.002076 1885944 gpu_resources.cc:119] Please NOTE: device: 0, GPU Compute Capability: 7.5, Driver API Version: 12.4, Runtime API Version: 11.8
	W0318 18:01:18.002839 1885944 gpu_resources.cc:164] device: 0, cuDNN Version: 8.9.
	I0318 18:01:18.112126 1885944 pir_interpreter.cc:1531] pir interpreter is running by multi-thread mode ...
	PaddlePaddle works well on 1 GPU.
	PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.

若是出現 Illegal instruction (core dumped) 是一個非常典型的硬體指令集不相容報錯<br>
在 PaddlePaddle 的環境中，這通常是因為你的 CPU 不支援 AVX (Advanced Vector Extensions) 指令集，或者是你安裝的 Paddle 版本啟用了你的 CPU 無法執行的優化指令（如 AVX2 或 AVX512<br>

=>	你將無法直接使用 PP-OCRv5（因為它是 3.0 引擎，有用到AVX2指令）<br>
	paddlepaddle 3.0.0 之後的版本都需要使用到AVX2的指令，因此會報錯，<br>
    試到最後目前GPU版本只能裝到 paddlepaddle-gpu-3.0.0rc1(CPU太舊)<br>
	這邊的 paddlepaddle是引擎和 PaddleOCR要跑得版本無關(引擎和OCR版本是可以分開的)，<br>
    只是最好儘量可以是相同的版本比較不會呼叫錯誤<br>

### 2.Step 2: 安裝基礎依賴 (requirements.txt) 
若有下載Code可以做，不會改到CODE就手動繼續往下裝就好了，可以直接使用PIP安裝好的套件直接來推論和訓練
> pip install -r requirements.txt



### 查看全部套件的版本號
Linux
> pip list | grep -E "numpy|opencv|paddle"

>   
	(py309OCRv5) yolo@yolo-ESC700-G2:~/PaddleOCRv5_project$ pip list | grep -E "numpy|opencv|paddle"
	numpy                    2.0.2
	paddlepaddle-gpu         3.2.2

Window
> pip list | findstr "numpy opencv paddle"


### 3. 降級 Numpy (推薦使用 4.8 或 4.9 系列，穩定支援 Numpy 1.x)
PaddlePaddle 安裝好會直接裝了 Numpy最新版<br>
Numpy 在 2.0 版本進行了重大更新，許多底層 API 與舊版不相容，而 Paddle 3.0b1 編譯時是針對 Numpy 1.x。<br>

>
    ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    paddlepaddle-gpu 3.0.0b1 requires numpy<2.0,>=1.13, but you have numpy 2.2.6 which is incompatible.
    Successfully installed numpy-2.2.6

手動將 Numpy 裝回 1.x 系列的最新版本（例如 1.26.4）
> pip install "numpy<2.0"

### 降級 OpenCV (推薦使用 4.8 或 4.9 系列，穩定支援 Numpy 1.x)
這步可不做，基本上都會自動先裝好，若是版本不符，再來降版
> pip install "opencv-python<4.10" "opencv-python-headless<4.10"

### 4. PaddleOCR 3.3.2
> pip install paddleocr==3.3.2

### 5. PaddleX 3.3.11
> pip install paddlex==3.3.11


### CUDA 版本查詢<br>
> nvcc -V<br>
=>	<br>
	nvcc: NVIDIA (R) Cuda compiler driver<br>
	Copyright (c) 2005-2022 NVIDIA Corporation<br>
	Built on Wed_Sep_21_10:33:58_PDT_2022<br>
	Cuda compilation tools, release 11.8, V11.8.89<br>
	Build cuda_11.8.r11.8/compiler.31833905_0<br>

#### 驗證 PaddlePaddle 安裝是否成功
> python -c "import paddle; paddle.utils.run_check()"

>   
    Running verify PaddlePaddle program ...
	I0317 09:56:34.586944 958971 program_interpreter.cc:243] New Executor is Running.
	W0317 09:56:34.587253 958971 gpu_resources.cc:119] Please NOTE: device: 0, GPU Compute Capability: 7.5, Driver API Version: 12.4, Runtime API Version: 11.8
	W0317 09:56:34.588006 958971 gpu_resources.cc:164] device: 0, cuDNN Version: 8.7.
	I0317 09:56:34.800230 958971 interpreter_util.cc:648] Standalone Executor is Used.
	PaddlePaddle works well on 1 GPU.
	PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.

<br>
<br>

### 最後使用的全部套件版本
##### CPU
    (py309OCRppv5) F:\UserData\Larry\Documents\VSCode Project\Python\PaddleOCR-v5>pip list | findstr "numpy opencv paddle"
    numpy                 1.26.4
    opencv-contrib-python 4.10.0.84
    paddleocr             3.3.2
    paddlepaddle          3.2.2
    paddlex               3.3.11

##### GPU
    (py309OCRv5) yolo@yolo-ESC700-G2:~/PaddleOCRv5_project$ pip list | grep -E "numpy|opencv|paddle"
    numpy                    1.26.4
    opencv-contrib-python    4.10.0.84
    paddleocr                3.3.2
    paddlepaddle-gpu         3.0.0rc1
    paddlex                  3.3.11


### 分別查詢各套件版本
[python]<br>
python --version<br>
=>3.9.25<br>

[numpy]<br>
python -c "import numpy; print(numpy.__version__)"<br>
=>1.26.4<br>

[paddlepaddle]<br>
##### CPU WIN10(CPU版本可以支援到最新)
python -c "import paddle; print('paddle OK:', paddle.__version__)"<br>
=>paddle OK: 3.2.2<br>

##### GPU Linux
python -c "import paddle; print('paddle OK:', paddle.__version__)"<br>
=>paddle OK: 3.0.0rc1<br>

[opencv]<br>
python -c "import cv2; print('OpenCV 版本:', cv2.__version__)"<br>
##### CPU WIN10
=> 4.10.0<br>

##### GPU Linux
=>4.10.0<br>

<br>
<br>
<br>
<br>
