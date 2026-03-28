#app.py
import sys
import os
import cv2
import numpy as np
import csv
import math
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QSlider, QLineEdit, QFileDialog,
    QTabWidget, QCheckBox, QGroupBox, QSpinBox, QDoubleSpinBox,
    QMessageBox, QGridLayout, QSizePolicy, QDial, QColorDialog, QProgressBar,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QListWidget, QSplitter
)
from PyQt6.QtGui import QPixmap, QImage, QColor, QFont, QCursor, QWheelEvent, QMouseEvent, QPainter, QPen
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPointF, QRectF

# ========== 0. 基础配置与环境 ==========
def get_base_path():
    """
    获取基础路径。
    打包后：返回 exe 文件所在目录（用于查找外部 models/set 文件夹）
    开发时：返回脚本所在目录
    """
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))

def get_cascade_path():
    """获取 Haar Cascade 文件路径"""
    cascade_filename = 'haarcascade_frontalface_default.xml'
    base_path = get_base_path()
    # 优先级 1: exe 同级目录的 models 文件夹 (用户手动复制的场景)
    cascade_path = os.path.join(base_path, 'models', cascade_filename)
    if os.path.exists(cascade_path):
        return cascade_path
    # 优先级 2: 打包后的临时目录 (如果使用了 --add-data 打包了 models)
    if getattr(sys, 'frozen', False):
        meipass_path = sys._MEIPASS
        paths_to_try = [
            os.path.join(meipass_path, 'models', cascade_filename),
            os.path.join(meipass_path, 'cv2', 'data', cascade_filename),
            os.path.join(meipass_path, cascade_filename),
        ]
        for path in paths_to_try:
            if os.path.exists(path):
                return path
    # 优先级 3: 开发环境 - 使用 cv2.data 内置路径
    try:
        cascade_path = os.path.join(os.path.dirname(cv2.data.__file__), cascade_filename)
        if os.path.exists(cascade_path):
            return cascade_path
    except:
        pass
    return None

# 获取 cascade 文件路径
CASCADE_PATH = get_cascade_path()
os.environ["QT_LOGGING_RULES"] = "qt.fonts.debug=false"
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 抠图模型配置
MATTING_MODELS = {
    "modnet_photographic_portrait_matting": {
        "name": "modnet_photographic_portrait_matting",
        "input_name": "input",
        "input_size": (512, 512),
        "weights": "modnet_photographic_portrait_matting.onnx"
    },
    "hivision_modnet": {
        "name": "hivision_modnet",
        "input_name": "input",
        "input_size": (512, 512),
        "weights": "hivision_modnet.onnx"
    }
}

# ========== 人脸检测配置 (OpenCV Haar Cascade) ==========
FACE_DETECTION_CONFIG = {
    "confidence_threshold": 0.2,
    "nms_threshold": 0.4,
    "scale_factor": 1.1,
    "min_neighbors": 5,
    "min_size": (30, 30)
}

# 模型权重路径配置
BASE_DIR = get_base_path()
WEIGHTS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(WEIGHTS_DIR, exist_ok=True)
WEIGHTS = {}
for model_info in MATTING_MODELS.values():
    WEIGHTS[model_info["name"]] = os.path.join(WEIGHTS_DIR, model_info["weights"])

# ========== NMS 函数 ==========
def py_cpu_nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

# ========== OpenCV 官方人脸检测器 ==========
class OpenCV_FaceDetector:
    def __init__(self, confidence_threshold=0.2, nms_threshold=0.4):
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        base_path = get_base_path()
        cascade_path = os.path.join(base_path, 'models', 'haarcascade_frontalface_default.xml')
        print(f"🔍 [Detector] Cascade 路径：{cascade_path}")
        print(f"🔍 [Detector] 文件存在：{os.path.exists(cascade_path)}")
        if os.path.exists(cascade_path):
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                print(f"⚠ Cascade 文件加载失败（文件可能损坏）")
                self.face_cascade = None
            else:
                print(f"✅ Cascade 加载成功")
        else:
            print("⚠ models 目录未找到 cascade，尝试 cv2.data 备用路径")
            try:
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                if os.path.exists(cascade_path):
                    self.face_cascade = cv2.CascadeClassifier(cascade_path)
                    print(f"✅ 使用 cv2.data 备用路径：{cascade_path}")
                else:
                    self.face_cascade = None
                    print("⚠ cv2.data 也未找到 cascade 文件")
            except Exception as e:
                self.face_cascade = None
                print(f"⚠ Cascade 初始化异常：{e}")

    def detect_face(self, image):
        try:
            if self.face_cascade is None or self.face_cascade.empty():
                print("⚠ Cascade 未加载，使用中心裁剪作为备用")
                h, w = image.shape[:2]
                face_w = int(w * 0.4)
                face_h = int(h * 0.5)
                x = (w - face_w) // 2
                y = (h - face_h) // 3
                return {
                    "x": x, "y": y, "width": face_w, "height": face_h,
                    "center_x": x + face_w//2, "center_y": y + face_h//2,
                    "eye_y": y + face_h//3, "box": (x, y, x + face_w, y + face_h)
                }
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            if len(faces) == 0:
                print("⚠ 未检测到人脸，使用中心裁剪")
                h, w = image.shape[:2]
                face_w = int(w * 0.4)
                face_h = int(h * 0.5)
                x = (w - face_w) // 2
                y = (h - face_h) // 3
                return {
                    "x": x, "y": y, "width": face_w, "height": face_h,
                    "center_x": x + face_w//2, "center_y": y + face_h//2,
                    "eye_y": y + face_h//3, "box": (x, y, x + face_w, y + face_h)
                }
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            x2 = x + w
            y2 = y + h
            print(f"✅ 人脸检测成功：({x},{y},{x2},{y2})")
            return {
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "center_x": x + w//2,
                "center_y": y + h//2,
                "eye_y": y + h//3,
                "box": (x, y, x2, y2)
            }
        except Exception as e:
            print(f"⚠ 人脸检测异常：{e}")
            h, w = image.shape[:2]
            face_w = int(w * 0.4)
            face_h = int(h * 0.5)
            x = (w - face_w) // 2
            y = (h - face_h) // 3
            return {
                "x": x, "y": y, "width": face_w, "height": face_h,
                "center_x": x + face_w//2, "center_y": y + face_h//2,
                "eye_y": y + face_h//3, "box": (x, y, x + face_w, y + face_h)
            }

# ========== 创建检测器工厂函数 ==========
def create_face_detector():
    """创建 OpenCV 人脸检测器"""
    print(f"🔍 [Factory] 创建 OpenCV 人脸检测器")
    return OpenCV_FaceDetector(
        confidence_threshold=FACE_DETECTION_CONFIG["confidence_threshold"],
        nms_threshold=FACE_DETECTION_CONFIG["nms_threshold"]
    )

# ========== 智能构图 ==========
def intelligent_crop_id_photo(image, target_width, target_height,
                              face_detector=None, face_params=None,
                              manual_offset_x=0, manual_offset_y=0, manual_scale=1.0):
    default_face_params = {
        "eye_line_ratio": 0.45,
        "head_top_margin_ratio": 0.15,
        "face_width_ratio": 0.6
    }
    if face_params is None:
        face_params = default_face_params
    img_height, img_width = image.shape[:2]
    face_info = None
    if face_detector is not None:
        face_info = face_detector.detect_face(image[:, :, :3])
    if face_info is not None:
        face_center_x = face_info["center_x"] + manual_offset_x
        face_eye_y = face_info["eye_y"] + manual_offset_y
        face_width = face_info["width"] * manual_scale
        face_height = face_info["height"] * manual_scale
        eye_line_ratio = face_params["eye_line_ratio"]
        head_top_margin_ratio = face_params["head_top_margin_ratio"]
        face_width_ratio = face_params["face_width_ratio"]
        target_face_width = target_width * face_width_ratio
        scale = target_face_width / face_width
        crop_height = int(target_height / scale)
        crop_width = int(target_width / scale)
        eye_target_y_in_crop = int(crop_height * eye_line_ratio)
        crop_y1 = face_eye_y - eye_target_y_in_crop
        head_top_margin_pixels = int(crop_height * head_top_margin_ratio)
        face_top_y = face_eye_y - int(face_height * 0.4)
        expected_crop_y1 = face_top_y - head_top_margin_pixels
        if crop_y1 > expected_crop_y1:
            crop_y1 = expected_crop_y1
        crop_y1 = max(0, crop_y1)
        crop_center_x = face_center_x
        crop_center_y = crop_y1 + crop_height // 2
        crop_x1 = max(0, crop_center_x - crop_width // 2)
        crop_x2 = min(img_width, crop_x1 + crop_width)
        crop_y2 = min(img_height, crop_y1 + crop_height)
        if crop_x2 - crop_x1 < crop_width:
            crop_x1 = max(0, crop_x2 - crop_width)
        if crop_y2 - crop_y1 < crop_height:
            crop_y1 = max(0, crop_y2 - crop_height)
        crop_x1 = int(crop_x1)
        crop_y1 = int(crop_y1)
        crop_x2 = int(crop_x2)
        crop_y2 = int(crop_y2)
    else:
        crop_width = min(img_width, int(img_height * target_width / target_height))
        crop_height = min(img_height, int(img_width * target_height / target_width))
        crop_x1 = (img_width - crop_width) // 2
        crop_y1 = (img_height - crop_height) // 2
        crop_x2 = crop_x1 + crop_width
        crop_y2 = crop_y1 + crop_height
    cropped_img = image[crop_y1:crop_y2, crop_x1:crop_x2]
    result_img = cv2.resize(
        cropped_img,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA
    )
    if len(result_img.shape) == 3 and result_img.shape[2] == 3:
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2BGRA)
    return result_img, face_info

# ========== 多模型兼容的抠图实现 ==========
class MattingModel:
    def __init__(self, model_config):
        self.session = None
        self.model_config = model_config
        self.model_path = model_config["weights_path"]
        self.input_name = model_config["input_name"]
        self.input_size = model_config["input_size"]

    def load_model(self):
        try:
            import onnxruntime as ort
            providers = ['CPUExecutionProvider']
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.insert(0, 'CUDAExecutionProvider')
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            inputs = self.session.get_inputs()
            if inputs:
                self.input_name = inputs[0].name
                input_shape = inputs[0].shape
                if len(input_shape) == 4:
                    h, w = input_shape[2], input_shape[3]
                    try:
                        h_int = int(h) if not isinstance(h, str) else -1
                        w_int = int(w) if not isinstance(w, str) else -1
                        if h_int > 0 and w_int > 0:
                            self.input_size = (w_int, h_int)
                    except (ValueError, TypeError) as e:
                        print(f"形状解析异常：{e}")
            return True
        except Exception as e:
            print(f"加载模型失败：{e}")
            return False

    def preprocess(self, image):
        if len(image.shape) == 3 and image.shape[2] == 4:
            img = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif len(image.shape) == 2:
            img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            img = image.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_input = cv2.resize(img, self.input_size, interpolation=cv2.INTER_AREA)
        img_input = img_input.astype(np.float32) / 255.0
        img_input = np.transpose(img_input, (2, 0, 1))
        img_input = np.expand_dims(img_input, axis=0)
        return img_input

    def postprocess(self, output, original_size):
        matte = np.squeeze(output)
        if len(matte.shape) > 2:
            matte = matte[0]
        matte = cv2.resize(matte, (original_size[1], original_size[0]), interpolation=cv2.INTER_AREA)
        matte = np.clip(matte, 0, 1) * 255
        return matte.astype(np.uint8)

    def predict(self, image):
        if self.session is None:
            if not self.load_model():
                return None
        original_size = (image.shape[0], image.shape[1])
        try:
            img_input = self.preprocess(image)
            output = self.session.run(None, {self.input_name: img_input})[0]
            matte = self.postprocess(output, original_size)
            result = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
            result[:, :, :3] = image[:, :, :3] if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            result[:, :, 3] = matte
            return result
        except Exception as e:
            print(f"推理失败：{e}")
            return None

def extract_human(image, model_key="modnet_photographic_portrait_matting"):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if model_key not in MATTING_MODELS:
        model_key = "modnet_photographic_portrait_matting"
    model_info = MATTING_MODELS[model_key]
    model_name = model_info["name"]
    model_path = WEIGHTS.get(model_name, WEIGHTS["modnet_photographic_portrait_matting"])
    if not os.path.exists(model_path):
        fallback_found = False
        for key, info in MATTING_MODELS.items():
            p = WEIGHTS.get(info["name"])
            if p and os.path.exists(p):
                model_path = p
                model_info = info
                fallback_found = True
                break
        if not fallback_found:
            return simple_matting(image)
    model_config = {
        "weights_path": model_path,
        "input_name": model_info["input_name"],
        "input_size": model_info["input_size"]
    }
    matting_model = MattingModel(model_config)
    result = matting_model.predict(image)
    if result is None:
        result = simple_matting(image)
    return result

def simple_matting(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    result = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    result[:, :, :3] = image[:, :, :3] if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    result[:, :, 3] = mask
    return result

# ========== 工具函数 ==========
def read_image(image_path):
    try:
        from PIL import Image
        img_pil = Image.open(image_path).convert("RGBA")
        img = np.array(img_pil)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        return img
    except:
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        return img

def save_image(img, save_path, dpi=300):
    if img is None:
        return False
    try:
        from PIL import Image
        if not isinstance(img, np.ndarray):
            return False
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            pil_img = Image.fromarray(img_rgba, mode='RGBA')
        elif len(img.shape) == 3 and img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb, mode='RGB')
        elif len(img.shape) == 2:
            pil_img = Image.fromarray(img, mode='L')
        else:
            return False
        save_path_lower = save_path.lower()
        if save_path_lower.endswith(('.jpg', '.jpeg')):
            if pil_img.mode == 'RGBA':
                background = Image.new('RGB', pil_img.size, (255, 255, 255))
                background.paste(pil_img, mask=pil_img.split()[3])
                pil_img = background
            pil_img.save(save_path, quality=95, dpi=(dpi, dpi))
        elif save_path_lower.endswith('.png'):
            pil_img.save(save_path, dpi=(dpi, dpi))
        elif save_path_lower.endswith('.bmp'):
            pil_img.save(save_path)
        else:
            if not save_path_lower.endswith('.png'):
                save_path = save_path + '.png'
            pil_img.save(save_path, dpi=(dpi, dpi))
        return True
    except Exception as e:
        print(f"保存失败：{e}")
        return False

def rotate_bound(image: np.ndarray, angle: float, center=None):
    (h, w) = image.shape[:2]
    if center is None:
        (cX, cY) = (w / 2, h / 2)
    else:
        (cX, cY) = center
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    rotated = cv2.warpAffine(image, M, (nW, nH))
    return rotated

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        return (255, 255, 255)
    return (
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16)
    )

def rgb_to_hex(rgb):
    return '#{:02X}{:02X}{:02X}'.format(rgb[0], rgb[1], rgb[2])

def convert_unit(value, unit, dpi=300):
    if unit == "毫米":
        return int(value * dpi / 25.4)
    return int(value)

def add_background(img, bg_rgb, kb_limit=0):
    if img is None or len(img.shape) != 3 or img.shape[2] != 4:
        return img
    b, g, r, a = cv2.split(img)
    alpha = a / 255.0
    bg = np.zeros_like(img[:, :, :3])
    bg[:, :, 0] = bg_rgb[2]
    bg[:, :, 1] = bg_rgb[1]
    bg[:, :, 2] = bg_rgb[0]
    result = (img[:, :, :3] * alpha[:, :, np.newaxis] + bg * (1 - alpha)[:, :, np.newaxis])
    result = result.astype(np.uint8)
    if kb_limit > 0:
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, 95]
        result_encode = cv2.imencode('.jpg', result, encode_param)[1]
        size_kb = len(result_encode) / 1024
        if size_kb > kb_limit:
            scale = math.sqrt(kb_limit / size_kb)
            new_w = int(result.shape[1] * scale)
            new_h = int(result.shape[0] * scale)
            result = cv2.resize(result, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return result

def whitening(img, strength):
    if strength <= 0 or img is None:
        return img
    strength = strength / 100.0
    img = img.astype(np.float32)
    if len(img.shape) == 3:
        if img.shape[2] == 4:
            img[:, :, :3] = img[:, :, :3] * (1 + strength)
        else:
            img = img * (1 + strength)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def adjust_brightness_contrast(img, brightness=0, contrast=1.0):
    if img is None:
        return img
    img = img.astype(np.float32)
    img = img + brightness
    img = (img - 127.5) * contrast + 127.5
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def generate_layout_array(input_height, input_width, LAYOUT_WIDTH=1780, LAYOUT_HEIGHT=1280):
    cols = LAYOUT_WIDTH // input_width
    rows = LAYOUT_HEIGHT // input_height
    dx = (LAYOUT_WIDTH - cols * input_width) // (cols + 1)
    dy = (LAYOUT_HEIGHT - rows * input_height) // (rows + 1)
    typography_arr = []
    for i in range(rows):
        for j in range(cols):
            x = dx + j * (input_width + dx)
            y = dy + i * (input_height + dy)
            typography_arr.append([x, y])
    return typography_arr, 0

def generate_layout_image(input_image, typography_arr, typography_rotate=0,
                          width=413, height=295, crop_line=True,
                          LAYOUT_WIDTH=1780, LAYOUT_HEIGHT=1280):
    layout_img = np.ones((LAYOUT_HEIGHT, LAYOUT_WIDTH, 3), dtype=np.uint8) * 255
    for (x, y) in typography_arr:
        if y + height <= LAYOUT_HEIGHT and x + width <= LAYOUT_WIDTH:
            img_rotated = rotate_bound(input_image, typography_rotate) if typography_rotate != 0 else input_image
            if img_rotated.shape[0] != height or img_rotated.shape[1] != width:
                img_rotated = cv2.resize(img_rotated, (width, height), interpolation=cv2.INTER_AREA)
            layout_img[y:y + height, x:x + width, :] = img_rotated[:height, :width, :3]
    if crop_line:
        for (x, y) in typography_arr:
            if x > 0 and x < LAYOUT_WIDTH:
                cv2.line(layout_img, (x, 0), (x, LAYOUT_HEIGHT), (200, 200, 200), 1)
            if x + width < LAYOUT_WIDTH:
                cv2.line(layout_img, (x + width, 0), (x + width, LAYOUT_HEIGHT), (200, 200, 200), 1)
        for (x, y) in typography_arr:
            if y > 0 and y < LAYOUT_HEIGHT:
                cv2.line(layout_img, (0, y), (LAYOUT_WIDTH, y), (200, 200, 200), 1)
            if y + height < LAYOUT_HEIGHT:
                cv2.line(layout_img, (0, y + height), (LAYOUT_WIDTH, y + height), (200, 200, 200), 1)
    return layout_img

def load_official_configs():
    base_dir = get_base_path()
    size_config = {
        "一寸 35×25mm": (413, 295),
        "大一寸 48×33mm": (567,390),
        "二寸 49×35mm": (579,413),
        "大二寸 53×35mm": (626,413),
        "小一寸 32×22mm": (378,260),
        "小二寸 45×35mm": (531,413)
    }
    size_path = os.path.join(base_dir, "set", "size_list.csv")
    if os.path.exists(size_path):
        try:
            with open(size_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)
                size_config = {}
                for row in reader:
                    if len(row) >= 3:
                        size_config[row[0]] = (int(row[1]), int(row[2]))
        except Exception as e:
            print(f"⚠ 加载尺寸配置失败：{e}")
    color_config = {
        "标准蓝": "438edb",
        "标准红": "ff0000",
        "白色": "ffffff",
        "浅蓝": "00bff3",
        "蓝色": "007dff",
        "中国红": "e60000",
        "暗红": "dc0000",
        "深红": "cc0000",
        "浅灰色": "c0c0c0"
    }
    color_path = os.path.join(base_dir, "set", "color_list.csv")
    if os.path.exists(color_path):
        try:
            with open(color_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)
                color_config = {}
                for row in reader:
                    if len(row) >= 2:
                        color_config[row[0]] = row[1]
        except Exception as e:
            print(f"⚠ 加载颜色配置失败：{e}")
    layout_sizes = {
        "六寸": (1780, 1280),
        "五寸": (1500, 1050),
        "A4": (2970, 2100),
        "3R": (1020, 760),
        "4R": (1520, 1020)
    }
    return size_config, color_config, layout_sizes

# ========== 交互式构图预览组件 ==========
class InteractiveCropView(QGraphicsView):
    params_changed = pyqtSignal(float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setMouseTracking(True)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.image_item = None
        self.face_circle_item = None
        self.crop_rect_item = None
        self.original_image = None
        self.face_info = None
        self.manual_offset_x = 0
        self.manual_offset_y = 0
        self.manual_scale = 1.0
        self.is_dragging = False
        self.last_mouse_pos = None
        self.target_width = 413
        self.target_height = 295
        self.setStyleSheet("""
        QGraphicsView {
            border: 2px solid #4F83CE;
            border-radius: 8px;
            background-color: #f5f5f5;
        }
        """)

    def set_image(self, image, matting_image=None, face_info=None):
        self.original_image = image
        self.face_info = face_info
        self.image_item = None
        self.face_circle_item = None
        self.crop_rect_item = None
        self.scene.clear()
        if image is not None:
            if len(image.shape) == 3 and image.shape[2] == 4:
                img_rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            else:
                img_rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
            h, w = img_rgba.shape[:2]
            bytes_per_line = w * 4
            q_img = QImage(img_rgba.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888)
            pixmap = QPixmap.fromImage(q_img)
            self.image_item = QGraphicsPixmapItem(pixmap)
            self.scene.addItem(self.image_item)
            self.scene.setSceneRect(0, 0, w, h)
            self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            if face_info:
                self._draw_face_overlay(face_info)
            self._update_crop_preview()

    def _draw_face_overlay(self, face_info):
        if self.face_circle_item and self.face_circle_item.scene() == self.scene:
            self.scene.removeItem(self.face_circle_item)
            self.face_circle_item = None
        x, y, w, h = face_info["box"]
        center_x = face_info["center_x"]
        center_y = face_info["center_y"]
        radius = min(w, h) // 3
        self.face_circle_item = self.scene.addEllipse(
            center_x - radius, center_y - radius,
            radius * 2, radius * 2,
            QPen(QColor(255, 0, 0, 200), 2)
        )

    def _update_crop_preview(self):
        if self.crop_rect_item and self.crop_rect_item.scene() == self.scene:
            self.scene.removeItem(self.crop_rect_item)
            self.crop_rect_item = None
        if self.original_image is None or self.face_info is None:
            return
        img_h, img_w = self.original_image.shape[:2]
        face_center_x = self.face_info["center_x"] + self.manual_offset_x
        face_center_y = self.face_info["center_y"] + self.manual_offset_y
        face_width = self.face_info["width"] * self.manual_scale
        target_face_width = self.target_width * 0.6
        scale = target_face_width / face_width
        crop_height = int(self.target_height / scale)
        crop_width = int(self.target_width / scale)
        crop_x1 = max(0, int(face_center_x - crop_width // 2))
        crop_y1 = max(0, int(face_center_y - crop_height // 2))
        crop_x2 = min(img_w, crop_x1 + crop_width)
        crop_y2 = min(img_h, crop_y1 + crop_height)
        if crop_x2 - crop_x1 < crop_width:
            crop_x1 = max(0, crop_x2 - crop_width)
        if crop_y2 - crop_y1 < crop_height:
            crop_y1 = max(0, crop_y2 - crop_height)
        self.crop_rect_item = self.scene.addRect(
            crop_x1, crop_y1, crop_x2 - crop_x1, crop_y2 - crop_y1,
            QPen(QColor(0, 0, 255, 200), 3, Qt.PenStyle.DashLine)
        )

    def set_target_size(self, width, height):
        self.target_width = width
        self.target_height = height
        self._update_crop_preview()

    def wheelEvent(self, event: QWheelEvent):
        if self.face_info is None:
            super().wheelEvent(event)
            return
        delta = event.angleDelta().y()
        if delta > 0:
            self.manual_scale *= 1.05
        else:
            self.manual_scale /= 1.05
        self.manual_scale = max(0.5, min(2.0, self.manual_scale))
        self._update_crop_preview()
        self.params_changed.emit(self.manual_offset_x, self.manual_offset_y, self.manual_scale)
        event.accept()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self.face_info is not None:
            self.is_dragging = True
            self.last_mouse_pos = event.pos()
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.is_dragging and self.last_mouse_pos is not None:
            delta = event.pos() - self.last_mouse_pos
            view_transform = self.transform()
            scale_factor = view_transform.m11()
            self.manual_offset_x += delta.x() / scale_factor
            self.manual_offset_y += delta.y() / scale_factor
            self._update_crop_preview()
            self.params_changed.emit(self.manual_offset_x, self.manual_offset_y, self.manual_scale)
            self.last_mouse_pos = event.pos()
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = False
            self.last_mouse_pos = None
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def reset_params(self):
        self.manual_offset_x = 0
        self.manual_offset_y = 0
        self.manual_scale = 1.0
        self._update_crop_preview()
        self.params_changed.emit(0, 0, 1.0)

# ========== 处理线程 ==========
class IDPhotoProcessThread(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.size_config, self.color_config, self.layout_sizes = load_official_configs()

    def run(self):
        try:
            self.progress.emit(10)
            img = read_image(self.params["input_path"])
            if img is None:
                raise ValueError("无法读取图片文件")
            self.progress.emit(20)
            matting_image = extract_human(img, model_key=self.params["matting_model"])
            self.progress.emit(30)
            if self.params["custom_size"]:
                target_height = convert_unit(self.params["height_val"], self.params["height_unit"], self.params["dpi"])
                target_width = convert_unit(self.params["width_val"], self.params["width_unit"], self.params["dpi"])
            else:
                target_height, target_width = self.size_config[self.params["size_name"]]
            face_detector = create_face_detector()
            face_params = {
                "eye_line_ratio": self.params["eye_line_ratio"],
                "head_top_margin_ratio": self.params["head_top_margin_ratio"],
                "face_width_ratio": self.params["face_width_ratio"]
            }
            standard_img, face_info = intelligent_crop_id_photo(
                matting_image,
                target_width=target_width,
                target_height=target_height,
                face_detector=face_detector,
                face_params=face_params,
                manual_offset_x=self.params.get("manual_offset_x", 0),
                manual_offset_y=self.params.get("manual_offset_y", 0),
                manual_scale=self.params.get("manual_scale", 1.0)
            )
            hd_img, _ = intelligent_crop_id_photo(
                matting_image,
                target_width=target_width * 2,
                target_height=target_height * 2,
                face_detector=face_detector,
                face_params=face_params,
                manual_offset_x=self.params.get("manual_offset_x", 0),
                manual_offset_y=self.params.get("manual_offset_y", 0),
                manual_scale=self.params.get("manual_scale", 1.0)
            )
            self.progress.emit(40)
            if self.params.get("face_align", False) and face_info is not None:
                standard_img = rotate_bound(standard_img, self.params.get("face_rotate_angle", 0))
                hd_img = rotate_bound(hd_img, self.params.get("face_rotate_angle", 0))
            self.progress.emit(50)
            if self.params.get("beauty", False):
                standard_img = whitening(standard_img, self.params.get("whitening", 0))
                standard_img = adjust_brightness_contrast(
                    standard_img,
                    brightness=self.params.get("brightness", 0),
                    contrast=self.params.get("contrast", 1.0)
                )
                hd_img = whitening(hd_img, self.params.get("whitening", 0))
                hd_img = adjust_brightness_contrast(
                    hd_img,
                    brightness=self.params.get("brightness", 0),
                    contrast=self.params.get("contrast", 1.0)
                )
            self.progress.emit(60)
            bg_hex = self.params["custom_color"] if self.params["use_custom_color"] else self.color_config[self.params["color_name"]]
            bg_rgb = hex_to_rgb(bg_hex)
            standard_img_bg = add_background(standard_img, bg_rgb, self.params["kb_limit"])
            hd_img_bg = add_background(hd_img, bg_rgb, self.params["kb_limit"])
            self.progress.emit(70)
            layout_photo = None
            if self.params.get("generate_layout", False):
                layout_size_name = self.params.get("layout_size", "六寸")
                layout_width, layout_height = self.layout_sizes[layout_size_name]
                typography_arr, typography_rotate = generate_layout_array(
                    input_height=target_height,
                    input_width=target_width,
                    LAYOUT_WIDTH=layout_width,
                    LAYOUT_HEIGHT=layout_height
                )
                layout_photo = generate_layout_image(
                    input_image=standard_img_bg,
                    typography_arr=typography_arr,
                    typography_rotate=typography_rotate,
                    width=target_width,
                    height=target_height,
                    crop_line=True,
                    LAYOUT_WIDTH=layout_width,
                    LAYOUT_HEIGHT=layout_height
                )
            self.progress.emit(90)
            result = {
                "id_photo_png": standard_img,
                "id_photo_jpg": standard_img_bg,
                "hd_photo": hd_img_bg,
                "layout_photo": layout_photo,
                "dpi": self.params["dpi"],
                "size": (target_height, target_width),
                "face_info": face_info if face_info else {"rectangle": (0, 0, 0, 0), "roll_angle": 0},
                "matting_image": matting_image
            }
            self.progress.emit(100)
            self.finished.emit(result)
        except Exception as e:
            import traceback
            self.error.emit(f"处理失败：{str(e)}\n详细信息：{traceback.format_exc()}")

# ========== 批量处理线程 ==========
class BatchProcessThread(QThread):
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(int, int)

    def __init__(self, file_paths, output_dir, params):
        super().__init__()
        self.file_paths = file_paths
        self.output_dir = output_dir
        self.params = params
        self.size_config, self.color_config, self.layout_sizes = load_official_configs()

    def run(self):
        success_count = 0
        fail_count = 0
        total = len(self.file_paths)
        try:
            model_info = MATTING_MODELS[self.params["matting_model"]]
            model_path = WEIGHTS.get(model_info["name"], WEIGHTS["modnet_photographic_portrait_matting"])
            model_config = {
                "weights_path": model_path,
                "input_name": model_info["input_name"],
                "input_size": model_info["input_size"]
            }
            matting_model = MattingModel(model_config)
            matting_model.load_model()
            face_detector = create_face_detector()
        except Exception as e:
            self.progress.emit(0, total, f"模型加载失败：{e}")
            self.finished.emit(0, total)
            return
        for i, path in enumerate(self.file_paths):
            try:
                self.progress.emit(i + 1, total, os.path.basename(path))
                img = read_image(path)
                if img is None:
                    fail_count += 1
                    continue
                matting_image = matting_model.predict(img)
                if matting_image is None:
                    matting_image = simple_matting(img)
                if self.params["custom_size"]:
                    target_height = convert_unit(self.params["height_val"], self.params["height_unit"], self.params["dpi"])
                    target_width = convert_unit(self.params["width_val"], self.params["width_unit"], self.params["dpi"])
                else:
                    target_height, target_width = self.size_config[self.params["size_name"]]
                face_params = {
                    "eye_line_ratio": self.params["eye_line_ratio"],
                    "head_top_margin_ratio": self.params["head_top_margin_ratio"],
                    "face_width_ratio": self.params["face_width_ratio"]
                }
                standard_img, _ = intelligent_crop_id_photo(
                    matting_image,
                    target_width=target_width,
                    target_height=target_height,
                    face_detector=face_detector,
                    face_params=face_params,
                    manual_offset_x=0,
                    manual_offset_y=0,
                    manual_scale=1.0
                )
                if self.params.get("beauty", False):
                    standard_img = whitening(standard_img, self.params.get("whitening", 0))
                    standard_img = adjust_brightness_contrast(
                        standard_img,
                        brightness=self.params.get("brightness", 0),
                        contrast=self.params.get("contrast", 1.0)
                    )
                bg_hex = self.params["custom_color"] if self.params["use_custom_color"] else self.color_config[self.params["color_name"]]
                bg_rgb = hex_to_rgb(bg_hex)
                result_img = add_background(standard_img, bg_rgb, self.params["kb_limit"])
                base_name = os.path.splitext(os.path.basename(path))[0]
                save_path = os.path.join(self.output_dir, f"{base_name}_id.jpg")
                if save_image(result_img, save_path, self.params["dpi"]):
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                print(f"处理文件 {path} 失败：{e}")
                fail_count += 1
        self.finished.emit(success_count, fail_count)

# ==========  主 GUI 窗口 ==========
class HivisionIDPhotoGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self._set_window_icon()
        self.setWindowTitle("证件照生成工具 精简版")
        self.setGeometry(100, 100, 1100, 650)
        self.setMinimumSize(900, 650)
        self.size_config, self.color_config, self.layout_sizes = load_official_configs()
        self.input_path = None
        self.result_data = None
        self.matting_image = None
        self.face_info = None
        self.manual_offset_x = 0
        self.manual_offset_y = 0
        self.manual_scale = 1.0
        self.batch_file_paths = []
        self.init_ui()

    def _set_window_icon(self):
        from PyQt6.QtGui import QIcon
        icon_path = None
        if getattr(sys, 'frozen', False):
            meipass_icon = os.path.join(sys._MEIPASS, "IDPhotoTool.ico")
            if os.path.exists(meipass_icon):
                icon_path = meipass_icon
        if icon_path is None:
            local_icon = os.path.join(get_base_path(), "IDPhotoTool.ico")
            if os.path.exists(local_icon):
                icon_path = local_icon
        if icon_path is None:
            local_icon = os.path.join(get_base_path(), "IDPhotoTool.ico")
            if os.path.exists(local_icon):
                icon_path = local_icon
        if icon_path and os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

    def _apply_modern_styles(self):
        """应用现代化 UI 样式"""
        style_sheet = """
        QGroupBox {
            font-weight: bold;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            margin-top: 12px;
            padding-top: 10px;
            background-color: #fcfcfc;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 6px;
            color: #4F83CE;
            background-color: #fcfcfc;
        }
        QSpinBox, QComboBox, QLineEdit, QDoubleSpinBox {
            border: 1px solid #d0d0d0;
            border-radius: 4px;
            padding: 6px;
            background-color: white;
            selection-background-color: #4F83CE;
        }
        QSpinBox:focus, QComboBox:focus, QLineEdit:focus, QDoubleSpinBox:focus {
            border: 1px solid #4F83CE;
        }
        QSlider::groove:horizontal {
            border: 1px solid #d0d0d0;
            height: 8px;
            background: #f0f0f0;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            background: #4F83CE;
            border: 1px solid #3a6eb8;
            width: 18px;
            margin: -6px 0;
            border-radius: 9px;
        }
        QSlider::handle:horizontal:hover {
            background: #3a6eb8;
        }
        QLabel {
            color: #555555;
        }
        QCheckBox {
            color: #555555;
        }
        QTabWidget::pane {
            border: 1px solid #e0e0e0;
            border-radius: 4px;
        }
        QTabBar::tab {
            background-color: #f5f5f5;
            border: 1px solid #e0e0e0;
            border-bottom: none;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            padding: 8px 16px;
            margin-right: 2px;
        }
        QTabBar::tab:selected {
            background-color: white;
            border-bottom: 2px solid #4F83CE;
        }
        QTabBar::tab:hover:!selected {
            background-color: #e8e8e8;
        }
        """
        self.setStyleSheet(style_sheet)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self._apply_modern_styles()
        
        # 使用 QSplitter 实现可拖拽调整左右面板宽度
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(main_splitter)
        
        # ========== 左侧面板 ==========
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_widget.setMinimumWidth(576)  # 最小宽度
        left_widget.setMaximumWidth(800)  # 最大宽度
        main_splitter.addWidget(left_widget)
        
        self.btn_select = QPushButton("📁 选择图片")
        self.btn_select.clicked.connect(self.select_image)
        self.btn_select.setStyleSheet("""
            QPushButton { 
                background-color: #4F83CE; 
                color: white; 
                padding: 12px; 
                font-size: 15px; 
                border: none; 
                border-radius: 6px; 
            } 
            QPushButton:hover { 
                background-color: #3a6eb8; 
            }
        """)
        left_layout.addWidget(self.btn_select)
        
        self.label_selected = QLabel("未选择图片")
        self.label_selected.setStyleSheet("color: #666; margin: 5px 0; font-size: 13px;")
        left_layout.addWidget(self.label_selected)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)
        
        hint_label = QLabel("💡 提示：可在预览区拖动鼠标调整人脸位置，滚轮调整人脸大小")
        hint_label.setStyleSheet("color: #4F83CE; font-weight: bold; margin: 10px 0;")
        left_layout.addWidget(hint_label)
        
        tab_widget = QTabWidget()
        left_layout.addWidget(tab_widget)
        
        # ========== 基础设置标签页 ==========
        basic_tab = QWidget()
        basic_layout = QVBoxLayout(basic_tab)
        tab_widget.addTab(basic_tab, "基础设置")
        
        size_group = QGroupBox("证件照尺寸")
        size_layout = QVBoxLayout(size_group)
        basic_layout.addWidget(size_group)
        
        self.size_combo = QComboBox()
        self.size_combo.addItems(list(self.size_config.keys()))
        self.size_combo.currentTextChanged.connect(self.on_size_changed)
        size_layout.addWidget(self.size_combo)
        
        self.custom_size_check = QCheckBox("自定义尺寸")
        self.custom_size_check.stateChanged.connect(self.toggle_custom_size)
        size_layout.addWidget(self.custom_size_check)
        
        custom_size_layout = QGridLayout()
        size_layout.addLayout(custom_size_layout)
        custom_size_layout.addWidget(QLabel("高度:"), 0, 0)
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(1, 5000)
        self.height_spin.setValue(413)
        self.height_spin.setEnabled(False)
        custom_size_layout.addWidget(self.height_spin, 0, 1)
        self.height_unit = QComboBox()
        self.height_unit.addItems(["像素", "毫米"])
        self.height_unit.setEnabled(False)
        custom_size_layout.addWidget(self.height_unit, 0, 2)
        custom_size_layout.addWidget(QLabel("宽度:"), 1, 0)
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(1, 5000)
        self.width_spin.setValue(295)
        self.width_spin.setEnabled(False)
        custom_size_layout.addWidget(self.width_spin, 1, 1)
        self.width_unit = QComboBox()
        self.width_unit.addItems(["像素", "毫米"])
        self.width_unit.setEnabled(False)
        custom_size_layout.addWidget(self.width_unit, 1, 2)
        
        color_group = QGroupBox("背景颜色")
        color_layout = QVBoxLayout(color_group)
        basic_layout.addWidget(color_group)
        
        self.color_combo = QComboBox()
        self.color_combo.addItems(list(self.color_config.keys()))
        color_layout.addWidget(self.color_combo)
        
        self.custom_color_check = QCheckBox("自定义颜色")
        self.custom_color_check.stateChanged.connect(self.toggle_custom_color)
        color_layout.addWidget(self.custom_color_check)
        
        custom_color_layout = QHBoxLayout()
        color_layout.addLayout(custom_color_layout)
        self.hex_input = QLineEdit()
        self.hex_input.setPlaceholderText("#438edb")
        self.hex_input.setEnabled(False)
        self.hex_input.setMaxLength(7)
        custom_color_layout.addWidget(self.hex_input, 1)
        self.btn_color_picker = QPushButton("🎨 选择颜色")
        self.btn_color_picker.setEnabled(False)
        self.btn_color_picker.clicked.connect(self.open_color_picker)
        custom_color_layout.addWidget(self.btn_color_picker)
        
        model_group = QGroupBox("抠图模型")
        model_layout = QVBoxLayout(model_group)
        basic_layout.addWidget(model_group)
        
        self.matting_combo = QComboBox()
        self.matting_combo.addItems(list(MATTING_MODELS.keys()))
        model_layout.addWidget(self.matting_combo)
        
        # ========== 构图配置标签页 ==========
        face_tab = QWidget()
        face_layout = QVBoxLayout(face_tab)
        tab_widget.addTab(face_tab, "构图配置")
        
        layout_group = QGroupBox("证件照构图参数")
        layout_layout = QGridLayout(layout_group)
        face_layout.addWidget(layout_group)
        
        layout_layout.addWidget(QLabel("双眼线高度比例："), 0, 0)
        self.eye_line_spin = QDoubleSpinBox()
        self.eye_line_spin.setRange(0.3, 0.6)
        self.eye_line_spin.setSingleStep(0.01)
        self.eye_line_spin.setValue(0.45)
        layout_layout.addWidget(self.eye_line_spin, 0, 1)
        
        layout_layout.addWidget(QLabel("头顶留白比例："), 1, 0)
        self.head_margin_spin = QDoubleSpinBox()
        self.head_margin_spin.setRange(0.05, 0.4)
        self.head_margin_spin.setSingleStep(0.01)
        self.head_margin_spin.setValue(0.15)
        layout_layout.addWidget(self.head_margin_spin, 1, 1)
        
        layout_layout.addWidget(QLabel("人脸宽度比例："), 2, 0)
        self.face_width_spin = QDoubleSpinBox()
        self.face_width_spin.setRange(0.4, 0.8)
        self.face_width_spin.setSingleStep(0.01)
        self.face_width_spin.setValue(0.6)
        layout_layout.addWidget(self.face_width_spin, 2, 1)
        
        reset_btn = QPushButton("🔄 重置构图")
        reset_btn.clicked.connect(self.reset_crop_params)
        face_layout.addWidget(reset_btn)
        
        rotate_group = QGroupBox("人脸旋转调整")
        rotate_layout = QVBoxLayout(rotate_group)
        face_layout.addWidget(rotate_group)
        
        self.face_align_check = QCheckBox("启用旋转对齐")
        rotate_layout.addWidget(self.face_align_check)
        
        rotate_layout.addWidget(QLabel("旋转角度（度）："))
        self.face_rotate_dial = QDial()
        self.face_rotate_dial.setRange(-15, 15)
        self.face_rotate_dial.setValue(0)
        self.face_rotate_dial.setNotchesVisible(True)
        rotate_layout.addWidget(self.face_rotate_dial)
        
        self.face_rotate_label = QLabel("当前角度：0°")
        self.face_rotate_dial.valueChanged.connect(lambda v: self.face_rotate_label.setText(f"当前角度：{v}°"))
        rotate_layout.addWidget(self.face_rotate_label)
        
        # ========== 输出质量设置标签页 ==========
        advanced_tab = QWidget()
        advanced_layout = QVBoxLayout(advanced_tab)
        advanced_layout.setContentsMargins(10, 10, 10, 10)
        advanced_layout.setSpacing(15)
        tab_widget.addTab(advanced_tab, "输出质量设置")
        
        output_group = QGroupBox("输出规格")
        output_layout = QGridLayout(output_group)
        output_layout.setSpacing(10)
        output_layout.setContentsMargins(15, 15, 15, 15)
        advanced_layout.addWidget(output_group)
        
        dpi_label = QLabel("输出 DPI:")
        dpi_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 600)
        self.dpi_spin.setValue(300)
        self.dpi_spin.setSuffix(" DPI")
        output_layout.addWidget(dpi_label, 0, 0)
        output_layout.addWidget(self.dpi_spin, 0, 1)
        
        kb_label = QLabel("文件大小限制:")
        kb_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.kb_spin = QSpinBox()
        self.kb_spin.setRange(0, 2000)
        self.kb_spin.setValue(0)
        self.kb_spin.setSpecialValueText("无限制")
        self.kb_spin.setSuffix(" KB")
        output_layout.addWidget(kb_label, 1, 0)
        output_layout.addWidget(self.kb_spin, 1, 1)
        
        advanced_layout.addStretch()
        
        # ========== 美颜设置标签页 ==========
        beauty_tab = QWidget()
        beauty_layout = QVBoxLayout(beauty_tab)
        beauty_layout.setContentsMargins(10, 10, 10, 10)
        beauty_layout.setSpacing(15)
        tab_widget.addTab(beauty_tab, "美颜设置")
        
        self.beauty_check = QCheckBox("启用美颜功能")
        self.beauty_check.setStyleSheet("QCheckBox { font-weight: bold; font-size: 14px; color: #4F83CE; }")
        self.beauty_check.stateChanged.connect(self.toggle_beauty)
        beauty_layout.addWidget(self.beauty_check)
        
        self.beauty_group = QGroupBox("调节参数")
        beauty_group_layout = QGridLayout(self.beauty_group)
        beauty_group_layout.setSpacing(10)
        beauty_group_layout.setContentsMargins(15, 15, 15, 15)
        beauty_layout.addWidget(self.beauty_group)
        
        w_label = QLabel("美白强度:")
        w_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.whitening_slider = QSlider(Qt.Orientation.Horizontal)
        self.whitening_slider.setRange(0, 100)
        self.whitening_slider.setValue(0)
        self.whitening_slider.setEnabled(False)
        self.label_whitening_val = QLabel("0%")
        self.label_whitening_val.setFixedWidth(40)
        self.label_whitening_val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.whitening_slider.valueChanged.connect(lambda v: self.label_whitening_val.setText(f"{v}%"))
        beauty_group_layout.addWidget(w_label, 0, 0)
        beauty_group_layout.addWidget(self.whitening_slider, 0, 1)
        beauty_group_layout.addWidget(self.label_whitening_val, 0, 2)
        
        b_label = QLabel("亮度调节:")
        b_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(-50, 50)
        self.brightness_slider.setValue(0)
        self.brightness_slider.setEnabled(False)
        self.label_brightness_val = QLabel("0")
        self.label_brightness_val.setFixedWidth(40)
        self.label_brightness_val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.brightness_slider.valueChanged.connect(lambda v: self.label_brightness_val.setText(f"{v}"))
        beauty_group_layout.addWidget(b_label, 1, 0)
        beauty_group_layout.addWidget(self.brightness_slider, 1, 1)
        beauty_group_layout.addWidget(self.label_brightness_val, 1, 2)
        
        c_label = QLabel("对比度:")
        c_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(0, 20)
        self.contrast_slider.setValue(10)
        self.contrast_slider.setEnabled(False)
        self.label_contrast_val = QLabel("1.0")
        self.label_contrast_val.setFixedWidth(40)
        self.label_contrast_val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.contrast_slider.valueChanged.connect(lambda v: self.label_contrast_val.setText(f"{v/10:.1f}"))
        beauty_group_layout.addWidget(c_label, 2, 0)
        beauty_group_layout.addWidget(self.contrast_slider, 2, 1)
        beauty_group_layout.addWidget(self.label_contrast_val, 2, 2)
        
        beauty_layout.addStretch()
        
        # ========== 排版设置标签页 ==========
        layout_tab = QWidget()
        layout_tab_layout = QVBoxLayout(layout_tab)
        layout_tab_layout.setContentsMargins(10, 10, 10, 10)
        layout_tab_layout.setSpacing(15)
        tab_widget.addTab(layout_tab, "排版设置")
        
        self.layout_check = QCheckBox("生成排版照 (多寸照)")
        self.layout_check.setStyleSheet("QCheckBox { font-weight: bold; font-size: 14px; color: #4F83CE; }")
        self.layout_check.stateChanged.connect(self.toggle_layout)
        layout_tab_layout.addWidget(self.layout_check)
        
        self.layout_group = QGroupBox("排版规格")
        layout_group_layout = QVBoxLayout(self.layout_group)
        layout_group_layout.setContentsMargins(15, 15, 15, 15)
        layout_tab_layout.addWidget(self.layout_group)
        
        layout_combo_label = QLabel("选择纸张尺寸:")
        self.layout_combo = QComboBox()
        self.layout_combo.addItems(list(self.layout_sizes.keys()))
        self.layout_combo.setEnabled(False)
        self.layout_combo.setStyleSheet("QComboBox { padding: 6px; }")
        
        layout_group_layout.addWidget(layout_combo_label)
        layout_group_layout.addWidget(self.layout_combo)
        
        layout_hint = QLabel("💡 排版照将自动填充所选纸张大小")
        layout_hint.setStyleSheet("color: #999; font-size: 12px; margin-top: 5px;")
        layout_group_layout.addWidget(layout_hint)
        
        layout_tab_layout.addStretch()
        
        # ========== 批量处理标签页 ==========
        batch_tab = QWidget()
        batch_layout = QVBoxLayout(batch_tab)
        tab_widget.addTab(batch_tab, "批量处理")
        
        batch_hint = QLabel("💡 批量处理将使用当前构图设置，自动居中人脸，不应用手动构图。")
        batch_hint.setStyleSheet("color: #e67e22; font-weight: bold; margin: 5px 0;")
        batch_layout.addWidget(batch_hint)
        
        self.btn_select_batch = QPushButton("📂 选择多张图片")
        self.btn_select_batch.clicked.connect(self.select_batch_images)
        batch_layout.addWidget(self.btn_select_batch)
        
        self.batch_list_widget = QListWidget()
        self.batch_list_widget.setMinimumHeight(150)
        batch_layout.addWidget(self.batch_list_widget)
        
        batch_output_layout = QHBoxLayout()
        self.batch_output_dir = QLineEdit()
        self.batch_output_dir.setPlaceholderText("选择输出文件夹...")
        self.batch_output_dir.setReadOnly(True)
        batch_output_layout.addWidget(self.batch_output_dir)
        self.btn_select_output = QPushButton("📁 输出目录")
        self.btn_select_output.clicked.connect(self.select_output_dir)
        batch_output_layout.addWidget(self.btn_select_output)
        batch_layout.addLayout(batch_output_layout)
        
        self.batch_progress_bar = QProgressBar()
        self.batch_progress_bar.setVisible(False)
        batch_layout.addWidget(self.batch_progress_bar)
        
        self.batch_status_label = QLabel("就绪")
        batch_layout.addWidget(self.batch_status_label)
        
        self.btn_start_batch = QPushButton("🚀 开始批量处理")
        self.btn_start_batch.clicked.connect(self.start_batch_processing)
        self.btn_start_batch.setEnabled(False)
        batch_layout.addWidget(self.btn_start_batch)
        
        self.btn_generate = QPushButton("🚀 生成证件照")
        self.btn_generate.clicked.connect(self.generate_photo)
        self.btn_generate.setEnabled(False)
        left_layout.addWidget(self.btn_generate)
        
        # ========== 右侧预览面板 ==========
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        main_splitter.addWidget(right_widget)
        
        # 隐藏分割线 - 将 handle 宽度设为 0
        main_splitter.setStyleSheet("""
        QSplitter::handle {
            background-color: transparent;
            width: 0px;
            border: none;
        }
        """)
        
        # 设置初始比例（左侧占 40%，右侧占 60%）
        main_splitter.setStretchFactor(0, 2)
        main_splitter.setStretchFactor(1, 3)
        main_splitter.setCollapsible(0, False)  # 左侧不可折叠
        main_splitter.setCollapsible(1, False)  # 右侧不可折叠
        
        preview_tab = QTabWidget()
        right_layout.addWidget(preview_tab)
        
        # 交互式构图预览
        interactive_tab = QWidget()
        interactive_layout = QVBoxLayout(interactive_tab)
        preview_tab.addTab(interactive_tab, "构图预览")
        
        self.interactive_view = InteractiveCropView()
        self.interactive_view.params_changed.connect(self.on_crop_params_changed)
        interactive_layout.addWidget(self.interactive_view)
        
        params_info_label = QLabel("当前参数：偏移 (0, 0) | 缩放 1.0x")
        params_info_label.setStyleSheet("color: #666; font-size: 12px;")
        self.params_info_label = params_info_label
        interactive_layout.addWidget(params_info_label)
        
        # 原始图片预览
        original_tab = QWidget()
        original_layout = QVBoxLayout(original_tab)
        preview_tab.addTab(original_tab, "原始图片")
        
        self.label_original = QLabel("请选择图片")
        self.label_original.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_original.setStyleSheet("border: 1px solid #eee; min-height: 300px;")
        original_layout.addWidget(self.label_original)
        
        # 证件照预览
        id_photo_tab = QWidget()
        id_layout = QVBoxLayout(id_photo_tab)
        preview_tab.addTab(id_photo_tab, "证件照（标准）")
        
        self.label_id_photo = QLabel("生成结果将显示在这里")
        self.label_id_photo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_id_photo.setStyleSheet("border: 1px solid #eee; min-height: 300px;")
        id_layout.addWidget(self.label_id_photo)
        
        # 高清照预览
        hd_tab = QWidget()
        hd_layout = QVBoxLayout(hd_tab)
        preview_tab.addTab(hd_tab, "证件照（高清）")
        
        self.label_hd_photo = QLabel("高清照将显示在这里")
        self.label_hd_photo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_hd_photo.setStyleSheet("border: 1px solid #eee; min-height: 300px;")
        hd_layout.addWidget(self.label_hd_photo)
        
        # 排版照预览
        layout_tab_preview = QWidget()
        layout_preview_layout = QVBoxLayout(layout_tab_preview)
        preview_tab.addTab(layout_tab_preview, "排版照")
        
        self.label_layout_photo = QLabel("排版照将显示在这里")
        self.label_layout_photo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_layout_photo.setStyleSheet("border: 1px solid #eee; min-height: 300px;")
        layout_preview_layout.addWidget(self.label_layout_photo)
        
        # 保存按钮
        save_layout = QHBoxLayout()
        right_layout.addLayout(save_layout)
        
        self.btn_save_id = QPushButton("💾 保存标准照")
        self.btn_save_id.clicked.connect(lambda: self.save_photo("id"))
        self.btn_save_id.setEnabled(False)
        save_layout.addWidget(self.btn_save_id)
        
        self.btn_save_hd = QPushButton("💾 保存高清照")
        self.btn_save_hd.clicked.connect(lambda: self.save_photo("hd"))
        self.btn_save_hd.setEnabled(False)
        save_layout.addWidget(self.btn_save_hd)
        
        self.btn_save_layout = QPushButton("💾 保存排版照")
        self.btn_save_layout.clicked.connect(lambda: self.save_photo("layout"))
        self.btn_save_layout.setEnabled(False)
        save_layout.addWidget(self.btn_save_layout)

    def on_size_changed(self, size_name):
        if size_name in self.size_config:
            height, width = self.size_config[size_name]
            self.interactive_view.set_target_size(width, height)

    def on_crop_params_changed(self, offset_x, offset_y, scale):
        self.manual_offset_x = offset_x
        self.manual_offset_y = offset_y
        self.manual_scale = scale
        self.params_info_label.setText(
            f"当前参数：偏移 ({offset_x:.1f}, {offset_y:.1f}) | 缩放 {scale:.2f}x"
        )

    def reset_crop_params(self):
        self.interactive_view.reset_params()
        self.manual_offset_x = 0
        self.manual_offset_y = 0
        self.manual_scale = 1.0
        QMessageBox.information(self, "提示", "构图参数已重置")

    def toggle_custom_size(self, state):
        enabled = state == Qt.CheckState.Checked.value
        self.height_spin.setEnabled(enabled)
        self.height_unit.setEnabled(enabled)
        self.width_spin.setEnabled(enabled)
        self.width_unit.setEnabled(enabled)

    def toggle_custom_color(self, state):
        enabled = state == Qt.CheckState.Checked.value
        self.hex_input.setEnabled(enabled)
        self.btn_color_picker.setEnabled(enabled)
        if enabled:
            self.open_color_picker()

    def open_color_picker(self):
        current_hex = self.hex_input.text().strip()
        if current_hex:
            try:
                rgb = hex_to_rgb(current_hex)
                initial_color = QColor(rgb[0], rgb[1], rgb[2])
            except:
                initial_color = QColor(79, 131, 206)
        else:
            initial_color = QColor(79, 131, 206)
        color = QColorDialog.getColor(initial_color, self, "选择背景颜色")
        if color.isValid():
            hex_color = color.name().upper()
            self.hex_input.setText(hex_color)

    def toggle_beauty(self, state):
        enabled = state == Qt.CheckState.Checked.value
        self.beauty_group.setEnabled(enabled)
        self.beauty_check.setEnabled(True)

    def toggle_layout(self, state):
        enabled = state == Qt.CheckState.Checked.value
        self.layout_group.setEnabled(enabled)
        self.layout_check.setEnabled(True)

    def select_batch_images(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "选择多张图片", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_paths:
            self.batch_file_paths = file_paths
            self.batch_list_widget.clear()
            for path in file_paths:
                self.batch_list_widget.addItem(os.path.basename(path))
            self.btn_start_batch.setEnabled(True)
            self.batch_status_label.setText(f"已选择 {len(file_paths)} 张图片")

    def select_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if dir_path:
            self.batch_output_dir.setText(dir_path)

    def start_batch_processing(self):
        if not self.batch_file_paths:
            QMessageBox.warning(self, "警告", "请先选择图片！")
            return
        if not self.batch_output_dir.text():
            QMessageBox.warning(self, "警告", "请选择输出文件夹！")
            return
        if self.custom_size_check.isChecked():
            height_val = self.height_spin.value()
            width_val = self.width_spin.value()
            height_unit = self.height_unit.currentText()
            width_unit = self.width_unit.currentText()
            size_name = "自定义"
        else:
            size_name = self.size_combo.currentText()
            height_val, width_val = self.size_config[size_name]
            height_unit = "像素"
            width_unit = "像素"
        if self.custom_color_check.isChecked():
            color_name = "自定义"
            custom_color = self.hex_input.text().strip() or "#FFFFFF"
            use_custom_color = True
        else:
            color_name = self.color_combo.currentText()
            custom_color = self.color_config[color_name]
            use_custom_color = False
        params = {
            "size_name": size_name,
            "custom_size": self.custom_size_check.isChecked(),
            "height_val": height_val,
            "height_unit": height_unit,
            "width_val": width_val,
            "width_unit": width_unit,
            "color_name": color_name,
            "use_custom_color": use_custom_color,
            "custom_color": custom_color,
            "matting_model": self.matting_combo.currentText(),
            "dpi": self.dpi_spin.value(),
            "kb_limit": self.kb_spin.value(),
            "beauty": self.beauty_check.isChecked(),
            "whitening": self.whitening_slider.value(),
            "brightness": self.brightness_slider.value(),
            "contrast": self.contrast_slider.value() / 10,
            "eye_line_ratio": self.eye_line_spin.value(),
            "head_top_margin_ratio": self.head_margin_spin.value(),
            "face_width_ratio": self.face_width_spin.value(),
        }
        self.btn_start_batch.setEnabled(False)
        self.batch_progress_bar.setVisible(True)
        self.batch_progress_bar.setValue(0)
        self.batch_status_label.setText("处理中...")
        self.batch_thread = BatchProcessThread(self.batch_file_paths, self.batch_output_dir.text(), params)
        self.batch_thread.progress.connect(self.on_batch_progress)
        self.batch_thread.finished.connect(self.on_batch_finished)
        self.batch_thread.start()

    def on_batch_progress(self, current, total, filename):
        self.batch_progress_bar.setMaximum(total)
        self.batch_progress_bar.setValue(current)
        self.batch_status_label.setText(f"处理中：{filename} ({current}/{total})")

    def on_batch_finished(self, success, fail):
        self.btn_start_batch.setEnabled(True)
        self.batch_progress_bar.setVisible(False)
        self.batch_status_label.setText(f"完成！成功：{success}, 失败：{fail}")
        QMessageBox.information(self, "批量处理完成", f"成功处理 {success} 张图片\n失败 {fail} 张图片")

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.input_path = file_path
            self.label_selected.setText(f"已选择：{os.path.basename(file_path)}")
            img = read_image(file_path)
            if img is not None:
                self._show_image(self.label_original, img)
                self.btn_generate.setEnabled(True)
                self._preview_matting(img)

    def _preview_matting(self, img, re_detect=False):
        if re_detect:
            matting_image = self.matting_image
        else:
            matting_image = extract_human(img, model_key=self.matting_combo.currentText())
        self.matting_image = matting_image
        print(f"🔍 [GUI] 创建 OpenCV 人脸检测器")
        face_detector = create_face_detector()
        self.face_info = face_detector.detect_face(matting_image[:, :, :3])
        if self.face_info:
            print(f"✓ [GUI] OpenCV 人脸检测成功：{self.face_info['box']}")
        else:
            print("⚠ [GUI] OpenCV 人脸检测失败，将使用中心裁剪")
        self.interactive_view.set_image(
            matting_image,
            matting_image,
            self.face_info
        )
        size_name = self.size_combo.currentText()
        if size_name in self.size_config:
            height, width = self.size_config[size_name]
            self.interactive_view.set_target_size(width, height)

    def generate_photo(self):
        if not self.input_path:
            QMessageBox.warning(self, "警告", "请先选择图片！")
            return
        if self.custom_size_check.isChecked():
            height_val = self.height_spin.value()
            width_val = self.width_spin.value()
            height_unit = self.height_unit.currentText()
            width_unit = self.width_unit.currentText()
            size_name = "自定义"
        else:
            size_name = self.size_combo.currentText()
            height_val, width_val = self.size_config[size_name]
            height_unit = "像素"
            width_unit = "像素"
        if self.custom_color_check.isChecked():
            color_name = "自定义"
            custom_color = self.hex_input.text().strip() or "#FFFFFF"
            use_custom_color = True
        else:
            color_name = self.color_combo.currentText()
            custom_color = self.color_config[color_name]
            use_custom_color = False
        face_params = {
            "eye_line_ratio": self.eye_line_spin.value(),
            "head_top_margin_ratio": self.head_margin_spin.value(),
            "face_width_ratio": self.face_width_spin.value(),
            "face_align": self.face_align_check.isChecked(),
            "face_rotate_angle": self.face_rotate_dial.value()
        }
        params = {
            "input_path": self.input_path,
            "size_name": size_name,
            "custom_size": self.custom_size_check.isChecked(),
            "height_val": height_val,
            "height_unit": height_unit,
            "width_val": width_val,
            "width_unit": width_unit,
            "color_name": color_name,
            "use_custom_color": use_custom_color,
            "custom_color": custom_color,
            "matting_model": self.matting_combo.currentText(),
            "dpi": self.dpi_spin.value(),
            "kb_limit": self.kb_spin.value(),
            "beauty": self.beauty_check.isChecked(),
            "whitening": self.whitening_slider.value(),
            "brightness": self.brightness_slider.value(),
            "contrast": self.contrast_slider.value() / 10,
            "generate_layout": self.layout_check.isChecked(),
            "layout_size": self.layout_combo.currentText() if self.layout_check.isChecked() else "六寸",
            "manual_offset_x": self.manual_offset_x,
            "manual_offset_y": self.manual_offset_y,
            "manual_scale": self.manual_scale,
            **face_params
        }
        self.btn_generate.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.thread = IDPhotoProcessThread(params)
        self.thread.finished.connect(self.on_process_finished)
        self.thread.error.connect(self.on_process_error)
        self.thread.progress.connect(self.progress_bar.setValue)
        self.thread.start()

    def on_process_finished(self, result):
        self.result_data = result
        self._show_image(self.label_id_photo, result["id_photo_jpg"])
        self._show_image(self.label_hd_photo, result["hd_photo"])
        if result["layout_photo"] is not None:
            self._show_image(self.label_layout_photo, result["layout_photo"])
        self.btn_save_layout.setEnabled(True)
        self.btn_save_id.setEnabled(True)
        self.btn_save_hd.setEnabled(True)
        self.btn_generate.setEnabled(True)
        self.progress_bar.setVisible(False)

    def on_process_error(self, error_msg):
        QMessageBox.critical(self, "错误", error_msg)
        self.btn_generate.setEnabled(True)
        self.progress_bar.setVisible(False)

    def save_photo(self, photo_type):
        if not self.result_data:
            QMessageBox.warning(self, "警告", "暂无可保存的图片！")
            return
        filters = "JPG Files (*.jpg);;PNG Files (*.png);;All Files (*.*)"
        default_ext = ".jpg"
        file_path, _ = QFileDialog.getSaveFileName(
            self, f"保存{photo_type}照", f"./idphoto_{photo_type}{default_ext}", filters
        )
        if file_path:
            if photo_type == "id":
                if file_path.lower().endswith(".png"):
                    img = self.result_data["id_photo_png"]
                else:
                    img = self.result_data["id_photo_jpg"]
            elif photo_type == "hd":
                img = self.result_data["hd_photo"]
            elif photo_type == "layout":
                img = self.result_data["layout_photo"]
            else:
                QMessageBox.warning(self, "警告", "无效的图片类型！")
                return
            if img is None:
                QMessageBox.warning(self, "警告", "暂无排版照可保存！")
                return
            success = save_image(img, file_path, self.result_data["dpi"])
            if success:
                QMessageBox.information(self, "成功", f"{photo_type}照已保存至：{file_path}")
            else:
                QMessageBox.critical(self, "错误", "保存图片失败！")

    def _show_image(self, label, img):
        if img is None:
            label.setText("暂无图片")
            return
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        else:
            img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        h, w = img_rgba.shape[:2]
        bytes_per_line = w * 4
        q_img = QImage(img_rgba.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888)
        pixmap = QPixmap.fromImage(q_img)
        pixmap = pixmap.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        label.setPixmap(pixmap)

# ========== 程序入口 ==========
if __name__ == "__main__":
    app = QApplication(sys.argv)
    from PyQt6.QtGui import QIcon
    icon_path = None
    if getattr(sys, 'frozen', False):
        meipass_icon = os.path.join(sys._MEIPASS, "IDPhotoTool.ico")
        if os.path.exists(meipass_icon):
            icon_path = meipass_icon
    if icon_path is None:
        local_icon = os.path.join(get_base_path(), "IDPhotoTool.ico")
        if os.path.exists(local_icon):
            icon_path = local_icon
    if icon_path and os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    font = QFont()
    if sys.platform == "win32":
        font.setFamily("Microsoft YaHei")
    else:
        font.setFamily("PingFang SC" if sys.platform == "darwin" else "WenQuanYi Micro Hei")
    font.setPointSize(10)
    app.setFont(font)
    window = HivisionIDPhotoGUI()
    window.show()
    sys.exit(app.exec())