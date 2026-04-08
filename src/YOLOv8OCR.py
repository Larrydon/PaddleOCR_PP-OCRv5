# YOLOv8 + PaddleOCR 3.3.2 + 車牌透視校正
from sympy import false, true

# V5 Torch會和Paddle dll衝突(WinError 127) 改用 ONNX
# from ultralytics import YOLO
import onnxruntime as ort

# from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import os
import time


# ================== 設定 ==================
TEST_IMG_PATH = "./test_car.jpg"
DEBUG_DIR = "ocr_debug"


class Config:
    DEBUG_PRINT = False
    DEBUG_BatchRun_PRINT = False


class YOLOv8PoseONNX:
    def __init__(self, model_path):
        # 1. 建立推論 Session (完全不依賴 torch)
        # providers 可以根據設備調整：['CPUExecutionProvider'] 或 ['CUDAExecutionProvider']
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )

        # 💡 自動獲取 ONNX 定義的輸入尺寸 [batch, channel, height, width]
        input_shape = self.session.get_inputs()[0].shape
        self.imgsz_h = input_shape[2]  # 應該會拿到 640
        self.imgsz_w = input_shape[3]  # 應該會拿到 640

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [x.name for x in self.session.get_outputs()]
        # self.imgsz = 640  # 匯出時設定的大小

    def preprocess(self, img):
        # 影像縮放與格式轉換 (滿足 YOLOv8 要求)
        h, w = img.shape[:2]
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = cv2.resize(
            image, (self.imgsz_w, self.imgsz_h)
        )  # 注意 OpenCV 是 (寬, 高)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # HWC to CHW
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image

    def infer(self, img_path):
        orig_img = cv2.imread(img_path)
        input_data = self.preprocess(orig_img)

        # 2. 執行推論
        outputs = self.session.run(self.output_names, {self.input_name: input_data})

        # outputs[0] 通常是 (1, 56, 8400) 的 Tensor (包含 Box, Score, 17個 Keypoints)
        return outputs[0]


# 使用範例
# pose_model = YOLOv8PoseONNX("yolov8n-pose.onnx")
# results = pose_model.infer("test_car.jpg")


YOLO_CONF_THRESH = 0.5  # YOLOv8 車牌信心值過濾 0.5
OCR_SCORE_THRESH = 0.5  # OCR 信心值過濾 0.5
YOLO_PAD_RATIO = 0.3  # 車牌裁切安全擴大比例 0.3
ROTATE_ANGLES = [2, 10, 15, -10, -15]  # 可調旋轉補償角度
INTERSECTION_GOOD = 0.6

OCR_WIDTH = 320
OCR_HIGHT = 48


FONT_TEXT = "./NotoSansCJKtc-Regular.otf"
# DET_MODEL_PATH = "./models/ch_PP-OCRv3_det_slim2_infer"
REC_MODEL_PATH = os.path.abspath(
    "./models/TW_PP-OCRv5_rec_infer"
)  # "PP-OCRv5_mobile_rec"
os.makedirs(DEBUG_DIR, exist_ok=True)


# ================== 初始化模型 ==================
# model = YOLO("yolov8s_pose_license_plate.pt")
pose_model = YOLOv8PoseONNX("yolov8s_pose_license_plate.onnx")


from paddleocr import TextRecognition

pocr = TextRecognition(
    model_dir="./models/TW_PP-OCRv5_rec_infer"
)  # model_name="PP-OCRv5_mobile_rec"
# output = pocr.predict(input=TEST_IMG_PATH, batch_size=1)
# print("\n" + "=" * 30)
# print(f"--- PaddleOCR ---")
# print("正在單獨加載 TextRecognition ...")
# for res in output:
#     res.print()


# ================== 工具函數 ==================
def expand_box(x1, y1, x2, y2, pad_ratio, img_w, img_h):
    bw = x2 - x1
    bh = y2 - y1
    px = int(bw * pad_ratio)
    py = int(bh * pad_ratio)

    return (
        max(0, x1 - px),
        max(0, y1 - py),
        min(img_w, x2 + px),
        min(img_h, y2 + py),
    )


def order_points(pts):
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    return np.array(
        [
            pts[np.argmin(s)],
            pts[np.argmin(diff)],
            pts[np.argmax(s)],
            pts[np.argmax(diff)],
        ],
        dtype="float32",
    )


# 透視水平，拉伸比例
def perspective_correction(img, boxes, padding_ratio=0.05):
    if not boxes:
        return np.array(img)

    all_pts = np.concatenate([np.array(b) for b in boxes], axis=0)
    hull = cv2.convexHull(all_pts.astype(np.float32))
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)
    box = order_points(box)

    # --- 新增：頂點外擴邏輯 (Padding) ---
    # 計算矩形的寬高，用來決定外擴的絕對像素值
    w_rect = rect[1][0]
    h_rect = rect[1][1]
    if w_rect < h_rect:
        w_rect, h_rect = h_rect, w_rect

    print(f"👀 perspective_correction 的 padding_ratio: {padding_ratio}")
    offset_w = w_rect * padding_ratio  # 左右各往外推 5%
    offset_h = h_rect * padding_ratio  # 上下各往外推 5%

    # 重新定義目標尺寸 (Dst) 也要加上這些 Offset
    new_w = int(w_rect + 2 * offset_w)
    new_h = int(h_rect + 2 * offset_h)

    # 這裡的關鍵是：dst 的座標要從 (offset_w, offset_h) 開始
    # 這樣變換後的圖，四周就會留出我們預設的空白，不會切到字
    dst = np.array(
        [[0, 0], [new_w - 1, 0], [new_w - 1, new_h - 1], [0, new_h - 1]],
        dtype="float32",
    )

    # 我們保持原始 box 不變，但讓 dst 變大，並調整對齊位置
    # 或是更簡單的方法：直接修改 box 讓它往外擴
    M = cv2.getPerspectiveTransform(
        box,
        np.array(
            [
                [offset_w, offset_h],
                [new_w - offset_w - 1, offset_h],
                [new_w - offset_w - 1, new_h - offset_h - 1],
                [offset_w, new_h - offset_h - 1],
            ],
            dtype="float32",
        ),
    )

    warped = cv2.warpPerspective(np.array(img), M, (new_w, new_h))
    return warped


# 透視校正傾斜自動適應
def perspective_correction_skew_adaptive(img, boxes):
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    if not boxes:
        return img

    h, w = img.shape[:2]

    # --- 修正後的座標檢查與 Clip 邏輯 ---
    fixed_boxes = []
    for box in boxes:
        # 確保 box 是 numpy array 方便計算
        box_np = np.array(box, dtype="float32")
        # 限制座標不超出圖片邊界 (避免原本的 TypeError)
        box_np[:, 0] = np.clip(box_np[:, 0], 0, w - 1)
        box_np[:, 1] = np.clip(box_np[:, 1], 0, h - 1)
        fixed_boxes.append(box_np)

    # 1. 取得所有點的凸包 (Convex Hull)
    all_pts = np.concatenate([np.array(b) for b in boxes], axis=0).astype(np.float32)
    hull = cv2.convexHull(all_pts)

    # --- 關鍵修改區 ---
    # 使用 approxPolyDP 逼近多邊形，直到點數剩下 4 個
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)

    # 如果逼近出來不是 4 個點（可能是 5, 6 個），則強制取 4 個點的最小外接矩形作為保險
    if len(approx) == 4:
        box = approx.reshape(4, 2).astype(np.float32)
    else:
        # 保險機制：如果多邊形太複雜，改用更穩定的四點逼近
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)

    # 務必進行排序，確保 tl, tr, br, bl 順序正確
    box = order_points(box)

    # 2. 計算原始車牌的「理想」寬高
    (tl, tr, br, bl) = box
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    maxWidth = int(max(width_top, width_bottom))
    height_left = np.linalg.norm(tl - bl)
    height_right = np.linalg.norm(tr - br)
    maxHeight = int(max(height_left, height_right))

    # 3. 🎯 核心修正：計算「歪斜補償量」
    dynamic_width_padding_ratio = 0.1  # 增加到 10%
    dynamic_height_padding_ratio = 0.2  # 增加到 20%
    print(
        f"👀 perspective_correction_skew_adaptive 的 dynamic_width_padding_ratio: {dynamic_width_padding_ratio}"
    )
    print(
        f"👀 perspective_correction_skew_adaptive 的 dynamic_height_padding_ratio: {dynamic_height_padding_ratio}"
    )
    # 如果你發現還是會切到，建議把這兩個 padding 調大
    dynamic_pad_w = int(maxWidth * dynamic_width_padding_ratio)
    dynamic_pad_h = int(maxHeight * dynamic_height_padding_ratio)
    new_w = maxWidth + 2 * dynamic_pad_w
    new_h = maxHeight + 2 * dynamic_pad_h

    # 4. 設定目標座標 (Dst)
    dst = np.array(
        [
            [dynamic_pad_w, dynamic_pad_h],
            [new_w - dynamic_pad_w - 1, dynamic_pad_h],
            [new_w - dynamic_pad_w - 1, new_h - dynamic_pad_h - 1],
            [dynamic_pad_w, new_h - dynamic_pad_h - 1],
        ],
        dtype="float32",
    )

    # 5. 執行變換
    M = cv2.getPerspectiveTransform(box, dst)

    # 使用 BORDER_REPLICATE 填充那些「斜出去」後留下的空隙
    warped = cv2.warpPerspective(
        img, M, (new_w, new_h), borderMode=cv2.BORDER_REPLICATE
    )
    return warped


def perspective_correction_yolo(img, kpts, padding_ratio=0.05):
    """
    修改上一個 perspective_correction_skew_adaptive()函式
    直接使用 YOLO-POSE 頂點進行透視校正
    kpts: YOLO 輸出的 4 個關鍵點 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

    取消[歪斜補償量]，直接使用4頂點就不用再計算了
    畫質優化： 增加了 flags=cv2.INTER_CUBIC，這會讓旋轉拉直後的文字邊緣更平滑，不會有明顯的階梯狀鋸齒。
    使用 LANCZOS4 會顯著提升邊緣銳利度，解決黏合問題
    """
    if kpts is None or len(kpts) != 4:
        return np.array(img)

    # 1. 確保座標格式並進行排序 (tl, tr, br, bl)
    # 直接使用傳入的點，不再尋找凸包
    src = np.array(kpts, dtype="float32").reshape(4, 2)
    src = order_points(src)

    # 2. 計算原始車牌的參考寬高 (用來決定輸出畫布的大小)
    (tl, tr, br, bl) = src
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    maxWidth = max(int(width_top), int(width_bottom))

    height_left = np.linalg.norm(tl - bl)
    height_right = np.linalg.norm(tr - br)
    maxHeight = max(int(height_left), int(height_right))

    # 3. 計算 Padding (外擴像素值)
    print(f"👀 perspective_correction_yolo 的 padding_ratio: {padding_ratio}")
    offset_w = maxWidth * padding_ratio
    offset_h = maxHeight * padding_ratio

    # 重新定義目標畫布尺寸 (加上左右、上下兩邊的 Padding)
    new_w = int(maxWidth + 2 * offset_w)
    new_h = int(maxHeight + 2 * offset_h)

    # 4. 定義目標映射點 (Dst)
    # 將原始 4 個點映射到新畫布中「縮進去」的位置，讓四周留出空白
    dst = np.array(
        [
            [offset_w, offset_h],  # 左上
            [new_w - offset_w - 1, offset_h],  # 右上
            [new_w - offset_w - 1, new_h - offset_h - 1],  # 右下
            [offset_w, new_h - offset_h - 1],  # 左下
        ],
        dtype="float32",
    )

    # 5. 執行透視轉換
    M = cv2.getPerspectiveTransform(src, dst)

    # 使用 BORDER_REPLICATE 填充邊緣，避免校正後的邊角出現黑邊
    warped = cv2.warpPerspective(
        np.array(img),
        M,
        (new_w, new_h),  # 計算原始車牌的參考寬高 (用來決定輸出畫布的大小)
        flags=cv2.INTER_LANCZOS4,  # cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return warped


def perspective_correction_yolo_no_padding(img, kpts):
    """
    完全複製訓練時的 get_rotate_crop_image 邏輯
    """
    if kpts is None or len(kpts) < 4:
        return None

    # 確保是 float32 numpy array
    points = np.array(kpts, dtype="float32").reshape(4, 2)

    # 1. 頂點排序 (完全照搬訓練截圖)
    d = 0.0
    for index in range(-1, 3):
        d += (
            -0.5
            * (points[index + 1][1] + points[index][1])
            * (points[index + 1][0] - points[index][0])
        )

    if d < 0:
        tmp = points.copy()
        points[1], points[3] = tmp[3], tmp[1]

    try:
        # 2. 計算校正後的目標寬高 (這是為了取得原始比例)
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3]),
            )
        )
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2]),
            )
        )

        # 3. 執行透視變換 (Warp)
        pts_std = np.float32(
            [
                [0, 0],
                [img_crop_width, 0],
                [img_crop_width, img_crop_height],
                [0, img_crop_height],
            ]
        )

        M = cv2.getPerspectiveTransform(points, pts_std)

        dst_img = cv2.warpPerspective(
            np.array(img),
            M,
            (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_LANCZOS4,  # cv2.INTER_CUBIC,
        )

        # 4. 自動旋轉檢查
        h_c, w_c = dst_img.shape[:2]
        if h_c / w_c >= 1.5:
            dst_img = np.rot90(dst_img)

        # # 5. 強制拉伸 (與訓練時完全一樣的第二次 Resize)
        # # 即使有二次損補，但因為訓練時也是這樣產生的，模型已經習慣這種特徵
        # dst_img = cv2.resize(
        #     dst_img, (OCR_WIDTH, OCR_HIGHT), interpolation=cv2.INTER_LANCZOS4
        # )

        return get_adaptive_enhanced_img(dst_img)

    except Exception as e:
        print(f"❌ 校正同步失敗: {e}")
        return None


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    return cv2.LUT(image, table)


def get_adaptive_enhanced_img(dst_img):
    # 1. 轉灰階來計算亮度
    gray = cv2.cvtColor(dst_img, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)  # 標準差代表對比度
    print(f"檢測 mean_brightness({mean_brightness:.1f})")
    print(f"檢測 std_brightness({std_brightness:.1f})")

    # 預設參數
    alpha = 1.1
    beta = 0

    # 2. 自動診斷邏輯
    if mean_brightness > 170:
        # 【太亮/過曝】: 不增加對比
        # #甚至稍微壓低亮度
        # alpha = 1.0
        # beta = -10
        # gamma = 0.5
        # print(f"🔆 檢測到高亮度 ({mean_brightness:.1f}), 跳過對比增強")
        print(f"🔆 檢測到高亮度 ({mean_brightness:.1f}), 跳過處理 🔆")
        return dst_img
    elif mean_brightness < 40:
        # 【暗】: 需要強力拉升
        alpha = 1.1
        beta = 100
        if std_brightness < 20:
            gamma = 2.0
        else:
            gamma = 1.6
        print(f"🔆 檢測到超低亮度 ({mean_brightness:.1f}), 啟動強力增強 🔆🔆🔆🔆")
    elif mean_brightness < 80:
        # 【太暗】: 需要微拉升就好
        alpha = 1.1
        beta = 100
        gamma = 1.8
        print(f"🔆 檢測到低亮度 ({mean_brightness:.1f}), 啟動強力增強 🔆🔆🔆")
    else:
        alpha = 1.3
        beta = 10
        gamma = 1.5
        print(f"🔆 檢測亮度正常 ({mean_brightness:.1f}), 使用標準增強  🔆🔆")

    # 3. 執行處理
    # 雙邊濾波，保留邊緣的平滑濾波器
    cvBilateral = cv2.bilateralFilter(dst_img, 9, 75, 75)

    # # 銳化卷積核
    # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    # sharpened = cv2.filter2D(cvBilateral, -1, kernel)

    # enhanced = cv2.convertScaleAbs(dst_img, alpha=alpha, beta=beta)
    # 試試看 gamma=0.8 或 0.5，會發現背景變白的速度比字體快，對比會更自然
    enhanced = adjust_gamma(cvBilateral, gamma)
    print(f"調整亮度 gamma({gamma:.1f})")

    gray2 = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    mean_brightness2 = np.mean(gray2)
    std_brightness2 = np.std(gray2)
    print(f"檢測 mean_brightness2({mean_brightness2:.1f})")
    print(f"檢測 std_brightness2({std_brightness2:.1f})")
    return enhanced


# YOLO 矩形 → 4點 polygon
def rect_to_poly(x1, y1, x2, y2):
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


# YOLO 矩形->強健邊緣檢測 → 4點 polygon
def find_plate_quad_robust(crop_img, idx=0):  # 記得加上 idx 參數
    # 1. 轉灰階並強化邊緣
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    # 使用中值濾波除去車牌上的螺絲或小雜訊，保留邊界線
    # blurred = cv2.medianBlur(gray, 9)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. Canny 邊緣檢測
    edged = cv2.Canny(blurred, 120, 255)

    # 3. 形態學處理 (核心修正：用閉合運算接斷線並過濾雜訊)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morphed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    # 4. 找最大輪廓
    cnts, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not cnts:
        return None

    # 找面積最大的輪廓
    c = max(cnts, key=cv2.contourArea)

    # 5. 取得凸包並逼近
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)

    # 稍微提高容忍度 (0.02 -> 0.03)，有助於減少多餘的點
    approx = cv2.approxPolyDP(hull, 0.03 * peri, True)

    # --- 修正：在判斷前先存下原始偵測點 (Raw Points) ---
    raw_debug = crop_img.copy()
    raw_pts = approx.reshape(-1, 2).astype(np.int32)

    # 畫出所有逼近點與連線
    cv2.polylines(raw_debug, [raw_pts], True, (255, 255, 255), 1)  # 白色細線

    for pt in raw_pts:
        cv2.circle(raw_debug, tuple(pt), 4, (255, 0, 0), -1)  # 藍色圓點

    cv2.putText(
        raw_debug,
        f"Approx count: {len(approx)}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 0),
        1,
    )

    cv2.imwrite(f"{DEBUG_DIR}/2_approx_raw_{idx}.jpg", raw_debug)
    debug_img = crop_img.copy()

    if len(approx) == 4:
        # --- 成功：畫綠色框 ---
        res_box = approx.reshape(4, 2).astype(np.int32)
        cv2.polylines(debug_img, [res_box], True, (0, 255, 0), 2)

        for pt in res_box:
            cv2.circle(debug_img, tuple(pt), 5, (0, 0, 255), -1)  # 紅色端點

        cv2.imwrite(f"{DEBUG_DIR}/2_find4point_{idx}.jpg", debug_img)
        return approx.reshape(4, 2).astype(np.float32)
    else:
        # --- 失敗處理：畫黃色保險框 ---
        # 關鍵修正：使用 hull (凸包) 算矩形，才不會被掉在中間的藍點拉扯
        rect = cv2.minAreaRect(hull)
        box_points = cv2.boxPoints(rect).astype(np.int32)

        # 畫出黃色框
        cv2.polylines(debug_img, [box_points], True, (0, 255, 255), 2)

        # 把失敗的藍點也補上去對照
        for pt in raw_pts:
            cv2.circle(debug_img, tuple(pt), 3, (255, 0, 0), -1)

        cv2.putText(
            debug_img,
            f"Failed: {len(approx)} pts",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )

        cv2.imwrite(f"{DEBUG_DIR}/2_fallback_rect_{idx}.jpg", debug_img)
        return box_points.astype(np.float32)


def save_debug_plate_image(crop_img, plate_box, idx, debug_dir):
    """
    在截圖上畫出偵測到的四個角點與邊框，並儲存 debug 影像。
    """

    # 1. 為了不破壞原始截圖，我們複製一份來畫圖
    # 如果 crop_img 是灰階，轉成 BGR 方便畫彩色線條
    if len(crop_img.shape) == 2:
        debug_canvas = cv2.cvtColor(crop_img, cv2.COLOR_GRAY2BGR)
    else:
        debug_canvas = crop_img.copy()

    # 2. 檢查 plate_box 是否有值
    if plate_box is not None:
        # 確保座標是整數，OpenCV 繪圖函數通常要求整數型態
        # plate_box 預期形狀為 (4, 2)
        pts = plate_box.astype(np.int32)

        # 3. 畫出四邊形的邊框 (綠色)
        # pts.reshape((-1, 1, 2)) 是 cv2.polylines 的標準格式

        cv2.polylines(
            debug_canvas,
            [pts.reshape((-1, 1, 2))],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2,
        )

        # 4. 畫出四個角點 (紅色圓點)
        # 順便標註 0, 1, 2, 3 序號，方便你檢查頂點順序（對透視變換很重要）
        for i, pt in enumerate(pts):
            cv2.circle(debug_canvas, (pt[0], pt[1]), 5, (0, 0, 255), -1)
            cv2.putText(
                debug_canvas,
                str(i),
                (pt[0] + 5, pt[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )

    # 5. 儲存結果
    output_path = f"{debug_dir}/2_find4point_{idx}.jpg"
    cv2.imwrite(output_path, debug_canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    print(f"Debug image saved to: {output_path}")


# 計算 polygon 交集比例
def intersection_ratio(ocr_box, yolo_rect):
    if len(ocr_box) < 4:
        return 0.0

    ocr_poly = np.array(ocr_box, dtype=np.float32)
    yolo_poly = rect_to_poly(*yolo_rect)
    area_ocr = cv2.contourArea(ocr_poly)

    if area_ocr <= 1:
        return 0.0

    inter_area, _ = cv2.intersectConvexConvex(ocr_poly, yolo_poly)
    return inter_area / area_ocr if area_ocr > 0 else 0.0


# === cv2.imread 等價處理 ===
def imread_equivalent(img_np):
    """
    將任意來源的 numpy image，強制轉成 PaddleOCR
    等價於 cv2.imread() 的輸入格式

    """

    # 1️⃣ 確保 numpy
    if not isinstance(img_np, np.ndarray):
        raise TypeError("input must be numpy array")

    # 2️⃣ dtype 強制 uint8
    if img_np.dtype != np.uint8:
        img_np = img_np.astype(np.uint8)

    # 3️⃣ channel 修正
    if img_np.ndim == 2:
        # gray → BGR
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    elif img_np.ndim == 3 and img_np.shape[2] == 3:
        pass
    else:
        raise ValueError(f"unsupported image shape: {img_np.shape}")

    # 4️⃣ 記憶體連續（⚠️ 非常關鍵）
    img_np = np.ascontiguousarray(img_np)

    # 5️⃣ 尺寸保護（rec 最低容忍）
    h, w = img_np.shape[:2]
    if h < 1 or w < 1:
        return None

    return img_np


def imread_equivalent_STRETCH_MODE(img_np):
    """
    強行拉伸到 192x48，完全對齊 A 模式訓練邏輯
    """
    # 確保格式
    if not isinstance(img_np, np.ndarray):
        raise TypeError("input must be numpy array")

    if img_np.dtype != np.uint8:
        img_np = img_np.astype(np.uint8)

    if img_np.ndim == 2:
        # gray → BGR
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    elif img_np.ndim == 3 and img_np.shape[2] == 3:
        pass
    else:
        raise ValueError(f"unsupported image shape: {img_np.shape}")

    img_np = np.ascontiguousarray(img_np)

    # 暴力拉伸：不理會比例，直接縮放到目標尺寸
    final_img = cv2.resize(
        img_np,
        (OCR_WIDTH, OCR_HIGHT),
        interpolation=cv2.INTER_LANCZOS4,  # cv2.INTER_CUBIC
    )

    return np.ascontiguousarray(final_img)


def parse_rec_result(ocr_res):
    """
    適用於 PaddleX 3.x (PP-OCRv5) TextRecognition 的輸出格式
    """
    # 1. 檢查輸入是否為空
    if not ocr_res:
        return "", 0.0

    # 2. 如果傳入的是 list (批次處理模式)，我們取第一個結果
    if isinstance(ocr_res, list):
        if len(ocr_res) == 0:
            return "", 0.0
        res = ocr_res[0]
    else:
        res = ocr_res

    # 3. 直接從字典中提取文字與評分
    try:
        # PaddleX V5 的 Key 名稱通常是 'rec_text' 與 'rec_score'
        # 使用 .get() 確保 Key 不存在時不會報錯
        text = res.get("rec_text", "")
        score = res.get("rec_score", 0.0)

        # 確保格式正確
        text = str(text).strip()
        score = float(score)

        return text, score

    except Exception as e:
        print(f"⚠️ 解析 V5 OCR 結果失敗: {e}, 原始數據: {res}")
        return "", 0.0


def rotate_image(img, angle):
    if abs(angle) < 1e-3:
        return img

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LANCZOS4,  # cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def ocr_det_valid(ocr_res, min_boxes=2):
    if not ocr_res or not ocr_res[0]:
        return False
    else:
        return True


def mean_intersection_ratio(ocr_boxes_global, yolo_rect):
    ratios = [intersection_ratio(box, yolo_rect) for box in ocr_boxes_global]
    return float(np.mean(ratios)) if ratios else 0.0


def draw_det_debug(pil_img, ocr_det_res, save_path):
    """
    將 OCR(det) 的 polygon 畫在 crop 圖上，輸出 debug 圖
    pil_img       : PIL.Image (crop_img)
    ocr_det_res   : PaddleOCR ocr(det=True) 回傳結果
    save_path    : debug 圖路徑
    """

    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    if not ocr_det_res or not ocr_det_res[0]:
        # 沒有任何 det 結果，直接存原圖
        draw.text((5, 5), "NO_OCR_DET", fill="red")
        img.save(save_path, quality=100)
        return

    for line in ocr_det_res[0]:
        box = line[0]  # 4 點 polygon

        # 將 polygon 畫成閉合線
        draw.line(box + [box[0]], fill="lime", width=2)
    img.save(save_path, quality=100)


def draw_text_with_outline(
    draw, position, text, font, text_color="yellow", outline_color="black"
):
    x, y = position
    # 畫四個角落的黑字
    for offset in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
        draw.text((x + offset[0], y + offset[1]), text, fill=outline_color, font=font)
    # 畫中間的亮色字
    draw.text((x, y), text, fill=text_color, font=font)


def save_padding_ocr_internal(img_np, save_path):
    """
    模擬 PaddleOCR 內部的預處理邏輯並存檔
    padding: True
    """
    h, w = img_np.shape[:2]
    target_h, target_w = (OCR_HIGHT, OCR_WIDTH)

    # 1. 算出等比例縮放後的寬度
    new_w = int(target_h * (w / h))

    # 2. 進行縮放 (高度固定 48)
    if new_w > target_w:
        # 如果太寬，強行縮到 target_w
        resized_img = cv2.resize(
            img_np, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4
        )
    else:
        # 如果寬度不足 target_w，等比例縮放後，右邊補黑邊 (Padding)
        temp_img = cv2.resize(
            img_np, (new_w, target_h), interpolation=cv2.INTER_LANCZOS4
        )
        resized_img = cv2.copyMakeBorder(
            temp_img, 0, 0, 0, target_w - new_w, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )

    cv2.imwrite(save_path, resized_img)
    return resized_img


def save_stretch_ocr_internal(img_np, save_path):
    """
    模擬『無 Padding、強行 Resize』的訓練預處理邏輯
    padding: False
    """

    # 直接強行拉伸，不計算比例，這才是『無 Padding』模式的真相
    resized_img = cv2.resize(
        img_np,
        (OCR_WIDTH, OCR_HIGHT),
        interpolation=cv2.INTER_LANCZOS4,  # cv2.INTER_LINEAR,  # PaddleOCR 預設通常用 Linear 或 Cubic
    )

    cv2.imwrite(save_path, resized_img)
    return resized_img


def DoOCR(
    orig_img,
    draw_result,
    idx,
    plate_box,
    debug_pose_img,
    debug_pose_draw,
    font_default,
    summary_results,
):
    bFoundLPR = True
    # --- 繪製 Pose Debug 資訊 ---
    pts = [(int(p[0]), int(p[1])) for p in plate_box]

    # 顏色定義：0:左上(紅), 1:右上(綠), 2:右下(藍), 3:左下(黃)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

    # --- 在 debug_pose_img 畫圖 ---
    if Config.DEBUG_PRINT and not Config.DEBUG_BatchRun_PRINT:
        for i, pt in enumerate(pts[:4]):
            r = 6
            debug_pose_draw.ellipse(
                (pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r),
                fill=colors[i],
                outline="white",
                width=2,
            )
        debug_pose_draw.line(pts + [pts[0]], fill="cyan", width=4)
        # 儲存這張「原圖標註 4 點」的 Debug 照片
        pose_debug_path = os.path.join(DEBUG_DIR, f"2_full_pose_detect_{idx}.jpg")
        debug_pose_img.save(pose_debug_path, quality=100)
        print(f"📍 原圖 4 點標註已另存至: {pose_debug_path}")

    # 執行透視校正
    orig_img_np = np.array(orig_img)
    try:
        # corrected = perspective_correction_yolo(orig_img_np, pts, 0.05)
        corrected = perspective_correction_yolo_no_padding(orig_img_np, pts)
    except Exception as e:
        print(f"校正發生錯誤：{e}，切換為原始裁切圖")
        return

    # 儲存校正/裁切後的 Debug 影像
    if Config.DEBUG_PRINT and not Config.DEBUG_BatchRun_PRINT:
        cv2.imwrite(f"{DEBUG_DIR}/3_perspective_{idx}.jpg", corrected)

    # 餵給 PaddleOCR REC
    final_ocr_res = imread_action_to_OCR(corrected, idx)
    plate_text, avg_score = parse_rec_result(final_ocr_res)

    # 存入摘要結果與畫圖
    pts_str = ", ".join([f"({p[0]},{p[1]})" for p in pts])
    summary_results.append(
        {"plate": plate_text, "score": avg_score, "coordinate": pts_str}
    )
    #
    # ---------- 畫回原圖 ----------
    if Config.DEBUG_PRINT:
        draw_ocr_debug(font_default, draw_result, plate_text, avg_score, pts)

    if not plate_text:
        bFoundLPR = False
    return summary_results, bFoundLPR


# ================== 主流程 ==================
# ================== 封裝後的辨識函式 ==================
def process_single_image(img_path, output_dir="results", debug=false):
    """
    輸入：圖片路徑, 輸出資料夾
    輸出：(結果列表, 儲存路徑)
    """
    Config.DEBUG_PRINT = debug
    start_time = time.time()

    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(img_path)
    final_output_path = os.path.join(output_dir, f"res_{filename}")

    # ⚠️ 修正點：必須使用傳入的 img_path，而不是 TEST_IMG_PATH
    try:
        orig_img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"無法開啟圖片 {img_path}: {e}")
        return [], None

    draw = ImageDraw.Draw(orig_img)
    img_W, img_H = orig_img.size

    # 準備存儲這張圖的所有結果
    summary_results = []

    try:
        font_default = ImageFont.truetype(FONT_TEXT, 20)
    except:
        font_default = ImageFont.load_default()

    """
    方法三:YOLO-Pose找出車牌4點    
    """
    # yolo_results = model.predict(img_path, verbose=False)
    # === 修改後的推論與解析邏輯 ===
    raw_output = pose_model.infer(img_path)  # 拿到 (1, 56, 8400)

    # ✨ 建立一張專門用來標註原圖點位的 debug 圖
    # 我們複製一份 orig_img，避免污染要給 OCR 顯示的 final_output
    debug_pose_img = orig_img.copy()
    debug_pose_draw = ImageDraw.Draw(debug_pose_img)

    # 1. 矩陣轉置與降維 -> (8400, 56)
    predictions = np.squeeze(raw_output).T

    # 2. 篩選信心值 (假設第 5 欄是 score)
    scores = predictions[:, 4]
    valid_mask = scores > YOLO_CONF_THRESH
    predictions = predictions[valid_mask]
    scores = scores[valid_mask]

    if len(predictions) == 0:
        print("🤷‍♂️ No detections found by ONNX model")
        return [], None

    # 3. 執行 NMS (避免同個車牌重複偵測)
    # YOLOv8-Pose 格式: [x_center, y_center, w, h, score, kpt1_x, kpt1_y, kpt1_conf, ...]
    boxes = predictions[:, :4]
    # 轉換為 [x1, y1, x2, y2]
    x1 = (boxes[:, 0] - boxes[:, 2] / 2) * (img_W / pose_model.imgsz_w)
    y1 = (boxes[:, 1] - boxes[:, 3] / 2) * (img_H / pose_model.imgsz_h)
    x2 = (boxes[:, 0] + boxes[:, 2] / 2) * (img_W / pose_model.imgsz_w)
    y2 = (boxes[:, 1] + boxes[:, 3] / 2) * (img_H / pose_model.imgsz_h)

    final_boxes = np.stack([x1, y1, x2, y2], axis=1)

    # 使用 OpenCV 的 NMS
    indices = cv2.dnn.NMSBoxes(
        final_boxes.tolist(), scores.tolist(), YOLO_CONF_THRESH, 0.45
    )

    # 4. 準備進行 OCR 的循環
    bFoundLPR = False
    # 修正處：確保 indices 是扁平的一維陣列
    if len(indices) > 0:
        # flatten() 可以相容新舊版本回傳的格式
        indices = np.array(indices).flatten()

        # 💡 新增：用來記錄「這是第幾個被認出的車牌」
        detected_count = 0

        for raw_idx in indices:
            # 現在 idx 直接就是整數索引了，不需要再做 i[0]

            # 提取 Bbox (使用原始索引 raw_idx)
            bx1, by1, bx2, by2 = map(int, final_boxes[raw_idx])
            crop_img = orig_img.crop((bx1, by1, bx2, by2))

            # 提取 Keypoints 並還原比例
            # 假設 YOLOv8-Pose 輸出格式：[x, y, w, h, conf, kpt1_x, kpt1_y, kpt1_conf, ...]
            # 我們要跳過前 5 欄
            kpts = predictions[raw_idx, 5:].reshape(-1, 3)
            plate_box = kpts[:4, :2].copy()  # 取前 4 點的 (x, y)

            # 座標還原回原圖比例 (ONNX 輸入 640x640)
            plate_box[:, 0] *= img_W / pose_model.imgsz_w
            plate_box[:, 1] *= img_H / pose_model.imgsz_h

            # 執行 DoOCR
            summary_results, bFoundLPR = DoOCR(
                orig_img,
                draw,
                detected_count,
                plate_box,
                debug_pose_img,
                debug_pose_draw,
                font_default,
                summary_results,
            )
            detected_count += 1

    # 絕對高度很小，可能是切好的車牌，直接OCR (例如高度小於 200 像素)
    is_likely_cropped_plate = (img_W <= OCR_WIDTH) and (img_H < 200)
    if is_likely_cropped_plate and not bFoundLPR:
        print(f"✨ YOLO-Post找不到4點，直接整張圖執行透視校正")

        x1, y1 = 0, 0
        x2, y2 = img_W, img_H
        plate_box = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],  # 左上  # 右上  # 右下  # 左下
            dtype=np.float32,
        )
        summary_results, bFoundLPR = DoOCR(
            orig_img,
            draw,
            0,
            plate_box,
            debug_pose_img,
            debug_pose_draw,
            font_default,
            summary_results,
        )

    # ================== 輸出 ==================
    if not bFoundLPR:
        print(f"🤷‍♂️ Cannot find any LPR 🤷‍♀️")

    if Config.DEBUG_PRINT:
        orig_img.save(final_output_path, quality=100)
        print(f"✅ 已輸出辨識結果：{final_output_path}")

    end_time = time.time()
    total_latency = end_time - start_time
    print(f"⏱️ 總處理耗時: {total_latency:.4f} 秒")  # 印出到小數點後四位

    # ⚠️ 關鍵：回傳 格式化後的清單 與 檔案路徑
    return summary_results, final_output_path


def imread_action_to_OCR(corrected, idx):
    # 歸一化
    ocr_input = imread_equivalent(corrected)
    # ocr_input = imread_equivalent_STRETCH_MODE(corrected)

    # 5. 模擬並儲存 OCR 內部「最終會變成什麼樣子」
    if Config.DEBUG_PRINT and not Config.DEBUG_BatchRun_PRINT:
        save_padding_ocr_internal(
            ocr_input, f"{DEBUG_DIR}/4.OCR_INTERNAL_PADDING_{idx}.jpg"
        )

    # return pocr.ocr(ocr_input, det=False, cls=False)
    return pocr.predict(ocr_input)


def draw_ocr_debug(font_default, draw, plate_text, avg_score, pts=None):
    # 如果 pts 為空或全 0 的處理邏輯
    if not pts or all(p == (0, 0) for p in pts):
        text_x, text_y = 0, 0
    else:
        # 取得多邊形的左上邊界點
        text_x = min(p[0] for p in pts)
        text_y = min(p[1] for p in pts)

        # ===== 根據 OCR 信心畫框顏色 =====
        if avg_score < OCR_SCORE_THRESH:
            draw.polygon(pts, outline="orange", width=3)
            draw.text(
                (text_x, max(0, text_y - 25)),  # 動態計算文字位置
                f"OCR low ({avg_score:.2f})",
                fill="yellow",
                font=font_default,
            )
        else:
            draw.polygon(pts, outline="red", width=3)
            draw.text(
                (text_x, max(0, text_y - 25)),  # 動態計算文字位置
                f"{plate_text} ({avg_score:.2f})",
                fill="yellow",
                font=font_default,
            )

        draw_text_with_outline(
            draw,
            (text_x, max(0, text_y - 25)),
            f"{plate_text} ({avg_score:.2f})",
            font_default,
        )

    pts_str = ", ".join([f"({p[0]},{p[1]})" for p in pts])
    print(f"✅ 辨識結果: {plate_text} | 置信度: {avg_score:.4f} | 座標: [{pts_str}]")


# 這裡保留一個測試區塊，當你直接執行 YOLOv8OCR.py 時才會跑
if __name__ == "__main__":
    test_res = process_single_image("./test_car.jpg", debug=True)
    print(f"測試結果: {test_res}")
