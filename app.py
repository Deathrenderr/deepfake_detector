import os
import io
import tempfile
import base64
import math
import traceback
from typing import List, Dict

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from flask import Flask, request, jsonify, render_template

# third-party helper to download Drive file
import gdown

# -------------------------
# CONFIG
# -------------------------
MODEL_URL = "https://drive.google.com/uc?id=13UCdd2OAek_pct3gNtdnG7r0ZerEkD1m"
CHECKPOINT_DIR = "checkpoints_v3_fixed"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "latest_checkpoint_v3.pth")
MODEL_NAME = "swin_base_patch4_window7_224"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# how many frames to sample from a video
DEFAULT_SAMPLE_FRAMES = 8
# how many frames to justify/explain (top k fake-conf frames)
EXPLAIN_TOP_K = 3

# -------------------------
# Flask app & Utils
# -------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")

# image transform (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Haar cascades for justification
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
EYE_CASCADE  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# -------------------------
# MODEL DOWNLOAD + LOAD
# -------------------------
def download_model_if_needed():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    if not os.path.exists(CHECKPOINT_PATH):
        print("⬇️ Downloading model weights from Google Drive...")
        gdown.download(MODEL_URL, CHECKPOINT_PATH, quiet=False)
        print("✅ Model downloaded.")
    else:
        print("⚙️ Checkpoint exists locally, skipping download.")

def build_model():
    print("🧠 Building model architecture...")
    model = timm.create_model(MODEL_NAME, pretrained=False)
    # replace classification head to 2 classes
    if hasattr(model, 'head') and isinstance(model.head, nn.Linear):
        model.head = nn.Linear(model.head.in_features, 2)
    elif hasattr(model, 'head') and hasattr(model.head, 'fc'):
        model.head.fc = nn.Linear(model.head.fc.in_features, 2)
    elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
        model.classifier = nn.Linear(model.classifier.in_features, 2)
    else:
        # fallback: find last linear and replace
        last_in = None
        for name, m in model.named_modules():
            if isinstance(m, nn.Linear):
                last_in = m.in_features
        if last_in is None:
            raise RuntimeError("Could not find a linear layer to replace in the model.")
        model.head = nn.Linear(last_in, 2)
    return model

def load_checkpoint_to_model(path):
    print(f"🔄 Loading weights from {path}")
    checkpoint = torch.load(path, map_location=DEVICE)
    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(DEVICE)
    model.eval()
    print("✅ Model loaded and ready.")
    return model

# initialize model at startup
download_model_if_needed()
model = load_checkpoint_to_model(CHECKPOINT_PATH)

# -------------------------
# PREDICTION & ANALYSIS
# -------------------------
def pil_from_bgr(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def tensor_from_pil(pil_img: Image.Image) -> torch.Tensor:
    return transform(pil_img).unsqueeze(0).to(DEVICE)

def predict_tensor(tensor: torch.Tensor):
    with torch.no_grad():
        out = model(tensor)  # assume shape [1,2] typically or [1,...]
        # handle possible extra spatial dims
        if out.ndim == 4 and out.shape[-1] == 2:
            out = out.mean(dim=(1,2))
        elif out.ndim == 3 and out.shape[-1] == 2:
            out = out.mean(dim=1)
        elif out.ndim > 2:
            out = out.view(out.size(0), -1)[:, :2]
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()
    # Model convention from your scripts: index 0 = FAKE, index 1 = REAL
    fake_conf = float(probs[0] * 100.0)
    real_conf = float(probs[1] * 100.0) if probs.shape[0] > 1 else float((1 - probs[0]) * 100.0)
    return fake_conf, real_conf, probs

# -------------------------
# JUSTIFICATION (adapted from prediction3.py)
# -------------------------
def laplacian_variance(img_gray):
    try:
        return float(cv2.Laplacian(img_gray, cv2.CV_64F).var())
    except Exception:
        return 0.0

def boundary_inconsistency(face_rgb):
    try:
        gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad = np.sqrt(gx**2 + gy**2)
        strip = int(max(1, min(face_rgb.shape[:2]) * 0.08))
        border_mean = np.mean([np.nanmean(grad[:strip, :]), np.nanmean(grad[-strip:, :]),
                               np.nanmean(grad[:, :strip]), np.nanmean(grad[:, -strip:])])
        interior_mean = np.nanmean(grad[strip:-strip, strip:-strip]) if grad.shape[0] > 2*strip else np.nanmean(grad)
        ratio = border_mean / (interior_mean + 1e-8)
        return float(min(200.0, ratio * 50.0))
    except Exception:
        return 0.0

def eye_hist_similarity(face_rgb, eyes):
    try:
        if len(eyes) < 2:
            return None
        h, w = face_rgb.shape[:2]
        def safe_crop(box):
            x, y, w_, h_ = map(int, box)
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x + w_), min(h, y + h_)
            if x2 <= x1 or y2 <= y1:
                return None
            return face_rgb[y1:y2, x1:x2]
        crop1, crop2 = safe_crop(eyes[0]), safe_crop(eyes[1])
        if crop1 is None or crop2 is None or crop1.size == 0 or crop2.size == 0:
            return None
        crop1 = cv2.resize(crop1, (32, 32)); crop2 = cv2.resize(crop2, (32, 32))
        g1 = cv2.cvtColor(crop1, cv2.COLOR_RGB2GRAY)
        g2 = cv2.cvtColor(crop2, cv2.COLOR_RGB2GRAY)
        h1 = cv2.calcHist([g1],[0],None,[64],[0,256]); h2 = cv2.calcHist([g2],[0],None,[64],[0,256])
        if h1.sum() == 0 or h2.sum() == 0:
            return None
        h1, h2 = h1.flatten()/(h1.sum()+1e-8), h2.flatten()/(h2.sum()+1e-8)
        sim = cv2.compareHist(h1.astype("float32"), h2.astype("float32"), cv2.HISTCMP_CORREL)
        if np.isnan(sim):
            return None
        return float((np.clip(sim, -1.0, 1.0)+1)/2*100)
    except Exception:
        return None

def justify_frame(frame_rgb: np.ndarray) -> List[str]:
    """
    Given a single RGB frame (numpy), run forensic heuristics and return textual explanations.
    """
    explanations = []
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))

    if len(faces) > 0:
        # choose largest face
        x, y, w, h = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]
        face_rgb = frame_rgb[y:y+h, x:x+w].copy()
        face_gray = gray[y:y+h, x:x+w].copy()

        lv = laplacian_variance(face_gray)
        texture_score = max(0.0, min(100.0, 200.0/(lv+1.0)))
        if texture_score > 70:
            explanations.append("The skin texture appears unnaturally smooth and lacks fine facial details.")
        elif texture_score > 40:
            explanations.append("Slight smoothness in texture, but mostly natural.")
        else:
            explanations.append("Texture appears detailed and realistic.")

        boundary_score = boundary_inconsistency(face_rgb)
        if boundary_score > 70:
            explanations.append("Strong edge inconsistencies suggest image blending or compositing.")
        elif boundary_score > 40:
            explanations.append("Minor boundary artifacts detected.")
        else:
            explanations.append("Facial boundaries are consistent with lighting and focus.")

        eyes = EYE_CASCADE.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=5, minSize=(10,10))
        if len(eyes) >= 2:
            eye_score = eye_hist_similarity(face_rgb, eyes)
            if eye_score is None:
                explanations.append("Eye symmetry skipped — unclear detection.")
            elif eye_score < 40:
                explanations.append("Eyes appear asymmetrical — possible sign of synthesis.")
            elif eye_score < 70:
                explanations.append("Eyes show mild variation, likely natural.")
            else:
                explanations.append("Eyes are symmetrical and realistic.")
        else:
            explanations.append("Eyes not detected clearly.")
    else:
        explanations.append("No face detected in frame — relying on model confidence alone.")

    return explanations

# -------------------------
# VIDEO ANALYSIS (frame sampling)
# -------------------------
def analyze_video_fileobj(fileobj: io.BytesIO, sample_frames: int = DEFAULT_SAMPLE_FRAMES) -> Dict:
    """
    Accepts BytesIO of uploaded video file. Samples frames evenly, predicts, and runs justification
    on top frames. Returns summary dict including thumbnails (base64).
    """
    # save to temp file (OpenCV needs a path)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(fileobj.read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        os.unlink(tmp_path)
        return {"error": "Could not open video file."}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    # if total_frames unknown or small, sample min(total_frames, sample_frames)
    if total_frames <= 0:
        # fallback: read sequentially and sample up to sample_frames evenly from what we can
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame.copy())
        cap.release()
        os.unlink(tmp_path)
        if len(frames) == 0:
            return {"error": "No frames found in video."}
        indices = np.linspace(0, len(frames)-1, num=min(sample_frames, len(frames)), dtype=int)
        sampled = [(int(i), frames[i]) for i in indices]
    else:
        # sample frames evenly across the video
        indices = np.linspace(0, total_frames-1, num=min(sample_frames, total_frames), dtype=int)
        sampled = []
        frame_idx = 0
        next_idx_ptr = 0
        target_idx = int(indices[next_idx_ptr]) if len(indices)>0 else None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if target_idx is not None and frame_idx == target_idx:
                sampled.append((int(frame_idx), frame.copy()))
                next_idx_ptr += 1
                if next_idx_ptr >= len(indices):
                    break
                target_idx = int(indices[next_idx_ptr])
            frame_idx += 1
        cap.release()
        os.unlink(tmp_path)

    # run model predictions on each sampled frame
    per_frame_results = []
    for idx, bgr in sampled:
        try:
            pil = pil_from_bgr(bgr)
            tensor = tensor_from_pil(pil)
            fake_conf, real_conf, _ = predict_tensor(tensor)
            # store as RGB for justification & thumbnail
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            per_frame_results.append({
                "frame_index": int(idx),
                "fake_conf": float(fake_conf),
                "real_conf": float(real_conf),
                "rgb": rgb  # keep for justification step (will remove before returning)
            })
        except Exception as e:
            print("Frame prediction error:", e)
            traceback.print_exc()

    if len(per_frame_results) == 0:
        return {"error": "No frames were analyzed successfully."}

    avg_fake = float(np.mean([f["fake_conf"] for f in per_frame_results]))
    avg_real = float(np.mean([f["real_conf"] for f in per_frame_results]))
    final_label = "FAKE" if avg_fake > avg_real else "REAL"

    # select top-k frames by fake confidence for justification
    per_frame_results_sorted = sorted(per_frame_results, key=lambda x: x["fake_conf"], reverse=True)
    explain_frames = per_frame_results_sorted[:EXPLAIN_TOP_K]

    explanations_out = []
    thumbs_out = []
    for entry in explain_frames:
        rgb = entry.pop("rgb")  # numpy rgb
        expl = justify_frame(rgb)
        # create thumbnail base64
        thumb = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (320, 180))
        _, buffer = cv2.imencode('.jpg', thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        thumb_b64 = base64.b64encode(buffer).decode('ascii')
        explanations_out.append({
            "frame_index": int(entry["frame_index"]),
            "fake_conf": float(entry["fake_conf"]),
            "real_conf": float(entry["real_conf"]),
            "explanations": expl,
            "thumbnail_b64": thumb_b64
        })

    return {
        "type": "video",
        "prediction": final_label,
        "avg_fake_confidence": avg_fake,
        "avg_real_confidence": avg_real,
        "num_sampled_frames": len(per_frame_results),
        "explain_frames": explanations_out
    }

# -------------------------
# IMAGE ANALYSIS (single image) - use justification + model
# -------------------------
def analyze_image_fileobj(fileobj: io.BytesIO) -> Dict:
    try:
        fileobj.seek(0)
        img = Image.open(fileobj).convert("RGB")
    except Exception:
        return {"error": "Could not read uploaded image."}
    rgb = np.array(img)
    tensor = tensor_from_pil(img)
    fake_conf, real_conf, _ = predict_tensor(tensor)
    label = "FAKE" if fake_conf > real_conf else "REAL"
    # get explanations for the image (full image)
    expl = justify_frame(rgb)
    # thumbnail
    thumb = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (320, 320))
    _, buffer = cv2.imencode('.jpg', thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    thumb_b64 = base64.b64encode(buffer).decode('ascii')

    return {
        "type": "image",
        "prediction": label,
        "fake_confidence": float(fake_conf),
        "real_confidence": float(real_conf),
        "explanations": expl,
        "thumbnail_b64": thumb_b64
    }

# -------------------------
# ROUTES
# -------------------------
@app.route('/')
def index():
    return render_template('index.html')  # will serve templates/index.html

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    f = request.files['file']
    if f.filename == '':
        return jsonify({"error": "Empty filename."}), 400

    # read into memory file-like
    file_bytes = io.BytesIO(f.read())
    # detect type via mime or filename extension
    filetype = f.content_type or ''
    lower = f.filename.lower()
    is_video = False
    if filetype.startswith('video') or lower.endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm')):
        is_video = True

    try:
        if is_video:
            result = analyze_video_fileobj(file_bytes)
        else:
            result = analyze_image_fileobj(file_bytes)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# -------------------------
# STARTUP
# -------------------------
if __name__ == "__main__":
    # if run directly, ensure model is loaded (already done above)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
