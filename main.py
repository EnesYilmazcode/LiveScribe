import time
import re
from PIL import Image
import cv2
import numpy as np
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

import os

print("Loading TrOCR model... this may take a moment.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

local_model_path = "./model"
hf_model_name = "microsoft/trocr-base-handwritten"

# Check if we have a saved model in ./model
# We look for config.json as a strong indicator that the model is there
if os.path.isdir(local_model_path) and os.path.exists(os.path.join(local_model_path, "config.json")):
    print(f"Found local model in {local_model_path}. Loading...")
    processor = TrOCRProcessor.from_pretrained(local_model_path)
    model = VisionEncoderDecoderModel.from_pretrained(local_model_path).to(device)
else:
    print(f"Local model not found. Downloading {hf_model_name}...")
    processor = TrOCRProcessor.from_pretrained(hf_model_name)
    model = VisionEncoderDecoderModel.from_pretrained(hf_model_name).to(device)
    
    print(f"Saving model to {local_model_path} for future runs...")
    if not os.path.exists(local_model_path):
        os.makedirs(local_model_path)
    processor.save_pretrained(local_model_path)
    model.save_pretrained(local_model_path)
    print("Model saved.")

# -------- helpers --------
def preprocess_canvas_to_binary(canvas_bgr):
    gray = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    _, binary = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary_clean = cv2.dilate(binary_clean, kernel, iterations=1)
    return binary_clean

def tight_crop_from_binary_on_canvas(canvas_bgr, binary_clean, pad=25):
    ys, xs = np.where(binary_clean > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(canvas_bgr.shape[1] - 1, x2 + pad)
    y2 = min(canvas_bgr.shape[0] - 1, y2 + pad)

    crop_bgr = canvas_bgr[y1:y2+1, x1:x2+1]
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(crop_rgb)

def normalize_expr(s: str) -> str:
    s = s.strip()
    s = s.replace("×", "*").replace("X", "*").replace("x", "*")
    s = s.replace("÷", "/").replace(":", "/")
    s = s.replace("−", "-").replace("—", "-").replace("–", "-")
    # Fix common misreadings of "+"
    s = s.replace("t", "+").replace("f", "+")
    s = re.sub(r"[^0-9\+\-\*\/\=\(\)\.\s]", "", s)
    if "=" in s:
        s = s.split("=")[0]
    return s.replace(" ", "")

def safe_eval(expr: str):
    if not expr:
        return None
    if re.fullmatch(r"[0-9\+\-\*\/\(\)\.]+", expr) is None:
        return None
    try:
        return eval(expr, {"__builtins__": {}})
    except Exception:
        return None

def trocr_read(pil_img):
    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        ids = model.generate(pixel_values, max_length=64)
    return processor.batch_decode(ids, skip_special_tokens=True)[0]

# -------- pad state --------
W, H = 900, 250
canvas = np.ones((H, W, 3), dtype=np.uint8) * 255

drawing = False
last_x, last_y = None, None

pen_thickness = 10
pen_color = (0, 0, 0)

# real-time OCR throttling
PREDICT_EVERY_SEC = 0.25   # run OCR at most 4x/sec
last_predict_time = 0.0
last_ocr = ""
last_expr = ""
last_ans = None

# helps avoid spamming OCR while you are actively drawing
last_stroke_time = 0.0
STROKE_IDLE_SEC = 0.12     # only run OCR if you haven't moved mouse for 120ms

def mouse_draw(event, x, y, flags, param):
    global drawing, last_x, last_y, canvas, last_stroke_time

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        last_x, last_y = x, y
        last_stroke_time = time.time()

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(canvas, (last_x, last_y), (x, y), pen_color, pen_thickness, lineType=cv2.LINE_AA)
            last_x, last_y = x, y
            last_stroke_time = time.time()

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        last_x, last_y = None, None
        last_stroke_time = time.time()


cv2.namedWindow("LiveScribe (Live OCR)")
cv2.setMouseCallback("LiveScribe (Live OCR)", mouse_draw)

print("Controls: draw with mouse | c=clear | s=save | q=quit")
print("Live OCR will update while you draw...")

while True:
    now = time.time()

    # Run OCR only if:
    # - enough time passed since last predict
    # - and you've been idle for a tiny moment
    if (now - last_predict_time) >= PREDICT_EVERY_SEC and (now - last_stroke_time) >= STROKE_IDLE_SEC:
        try:
            binary_clean = preprocess_canvas_to_binary(canvas)
            pil_crop = tight_crop_from_binary_on_canvas(canvas, binary_clean, pad=25)

            if pil_crop is None:
                last_ocr, last_expr, last_ans = "", "", None
            else:
                raw = trocr_read(pil_crop)
                expr = normalize_expr(raw)
                ans = safe_eval(expr)

                last_ocr, last_expr, last_ans = raw, expr, ans

            last_predict_time = now

        except Exception as e:
            # don't crash the live loop if OCR fails once
            last_ocr = f"(OCR error)"
            last_expr = ""
            last_ans = None
            last_predict_time = now

    # draw overlay text
    display = canvas.copy()
    cv2.putText(display, f"OCR: {last_ocr}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(display, f"Expr: {last_expr}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(display, f"Ans:  {last_ans}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("LiveScribe (Live OCR)", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("c"):
        canvas[:] = 255
        last_ocr, last_expr, last_ans = "", "", None
    elif key == ord("s"):
        cv2.imwrite("draw.png", canvas)
        print("Saved draw.png")

cv2.destroyAllWindows()
