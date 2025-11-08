import os, sys, traceback
import cv2
import numpy as np
from PIL import Image
import torch, torch.nn as nn
import timm
from torchvision import transforms

# ========== CONFIG ==========
CHECKPOINT = r"checkpoints_v3_fixed\epoch_15_swinv3.pth"
MODEL_NAME = "swin_base_patch4_window7_224"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FAST = False
DEBUG = True

# Haar cascades
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
EYE_CASCADE  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# ========== Debug Helper ==========
def dbg_print(*args, **kwargs):
    if DEBUG:
        print("[DEBUG]:", *args, **kwargs)

# ========== Model Loading ==========
def load_model():
    dbg_print("Creating model:", MODEL_NAME)
    model = timm.create_model(MODEL_NAME, pretrained=False)
    if hasattr(model, "head") and hasattr(model.head, "in_features"):
        model.head = nn.Linear(model.head.in_features, 2)
    elif hasattr(model, "classifier") and hasattr(model.classifier, "in_features"):
        model.classifier = nn.Linear(model.classifier.in_features, 2)
    else:
        for name, m in model.named_modules():
            if isinstance(m, nn.Linear):
                last = m.in_features
        model.head = nn.Linear(last, 2)

    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model = model.to(DEVICE).eval()
    dbg_print("✅ Model loaded and moved to", DEVICE)
    return model

# ========== Forensic Metrics ==========
def laplacian_variance(img_gray):
    try:
        return float(cv2.Laplacian(img_gray, cv2.CV_64F).var())
    except Exception as e:
        dbg_print("laplacian_variance error:", e)
        return 0.0

def boundary_inconsistency(face_rgb):
    try:
        gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad = np.sqrt(gx**2 + gy**2)
        strip = int(max(1, min(face_rgb.shape[:2]) * 0.08))
        if grad.size == 0:
            return 0.0
        border_mean = np.mean([
            np.nanmean(grad[:strip, :]), np.nanmean(grad[-strip:, :]),
            np.nanmean(grad[:, :strip]), np.nanmean(grad[:, -strip:])
        ])
        interior_mean = np.nanmean(grad[strip:-strip, strip:-strip]) if grad.shape[0] > 2*strip else np.nanmean(grad)
        ratio = border_mean / (interior_mean + 1e-8)
        return float(min(200.0, ratio * 50.0))
    except Exception as e:
        dbg_print("boundary_inconsistency error:", e)
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
        e1, e2 = eyes[0], eyes[1]
        crop1, crop2 = safe_crop(e1), safe_crop(e2)
        if crop1 is None or crop2 is None or crop1.size == 0 or crop2.size == 0:
            return None
        crop1 = cv2.resize(crop1, (32, 32)); crop2 = cv2.resize(crop2, (32, 32))
        g1 = cv2.cvtColor(crop1, cv2.COLOR_RGB2GRAY)
        g2 = cv2.cvtColor(crop2, cv2.COLOR_RGB2GRAY)
        h1 = cv2.calcHist([g1],[0],None,[64],[0,256])
        h2 = cv2.calcHist([g2],[0],None,[64],[0,256])
        if h1.sum() == 0 or h2.sum() == 0:
            return None
        h1, h2 = h1.flatten()/(h1.sum()+1e-8), h2.flatten()/(h2.sum()+1e-8)
        sim = cv2.compareHist(h1.astype("float32"), h2.astype("float32"), cv2.HISTCMP_CORREL)
        if np.isnan(sim):
            return None
        return float((np.clip(sim, -1.0, 1.0)+1)/2*100)
    except Exception as e:
        dbg_print("eye_hist_similarity error:", e)
        return None

# ========== Justification Function ==========
def justify_image(model, image_path):
    if not os.path.exists(image_path):
        print("❌ File not found:", image_path)
        return

    try:
        orig = Image.open(image_path).convert("RGB")
        img_np = np.array(orig)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    except Exception as e:
        print("❌ Could not process image:", e)
        if DEBUG: traceback.print_exc()
        return

    # Prediction (robust output shape handling)
    tensor = transform(orig).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(tensor)

        # Handle Swin spatial logits (e.g., [1,7,7,2])
        if out.ndim == 4 and out.shape[-1] == 2:
            dbg_print(f"Detected spatial output {out.shape}, averaging over HxW.")
            out = out.mean(dim=(1, 2))  # -> [1,2]
        elif out.ndim == 3 and out.shape[-1] == 2:
            dbg_print(f"Detected 3D output {out.shape}, averaging over tokens.")
            out = out.mean(dim=1)
        elif out.ndim > 2:
            dbg_print(f"Unexpected output shape {out.shape}, flattening.")
            out = out.view(out.size(0), -1)
            out = out[:, :2]

        probs = torch.softmax(out, dim=1)[0].cpu().numpy()

    dbg_print("Model output shape after processing:", probs.shape)
    if probs.shape[0] > 2:
        dbg_print("⚠️ Model has", probs.shape[0], "outputs; using first two as [FAKE, REAL].")
        probs = probs[:2]

    pred_idx = int(np.argmax(probs))
    conf = float(probs[pred_idx] * 100)

    label_map = {0: "FAKE", 1: "REAL"}
    pred_label = label_map.get(pred_idx, str(pred_idx))
    explanation = []

    # Forensics
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))
    if len(faces) > 0:
        x, y, w, h = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]
        face_rgb = img_np[y:y+h, x:x+w].copy()
        face_gray = gray[y:y+h, x:x+w].copy()

        lv = laplacian_variance(face_gray)
        texture_score = max(0.0, min(100.0, 200.0/(lv+1.0)))
        if texture_score > 70:
            explanation.append("The skin texture appears unnaturally smooth and lacks fine facial details.")
        elif texture_score > 40:
            explanation.append("Slight smoothness in texture, but mostly natural.")
        else:
            explanation.append("Texture appears detailed and realistic.")

        boundary_score = boundary_inconsistency(face_rgb)
        if boundary_score > 70:
            explanation.append("Strong edge inconsistencies suggest image blending or compositing.")
        elif boundary_score > 40:
            explanation.append("Minor boundary artifacts detected.")
        else:
            explanation.append("Facial boundaries are consistent with lighting and focus.")

        eyes = EYE_CASCADE.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=5, minSize=(10,10))
        if len(eyes) >= 2:
            eye_score = eye_hist_similarity(face_rgb, eyes)
            if eye_score is None:
                explanation.append("Eye symmetry skipped — unclear detection.")
            elif eye_score < 40:
                explanation.append("Eyes appear asymmetrical — possible sign of synthesis.")
            elif eye_score < 70:
                explanation.append("Eyes show mild variation, likely natural.")
            else:
                explanation.append("Eyes are symmetrical and realistic.")
        else:
            explanation.append("Eyes not detected clearly.")
    else:
        explanation.append("No face detected — relying solely on model confidence.")

    # Output
    print("\n🧠 DeepFake Forensic Report")
    print("--------------------------------")
    print(f"🧾 Prediction: {pred_label} ({conf:.2f}% confidence)")
    print("--------------------------------")
    for line in explanation:
        print("•", line)
    print("--------------------------------")
    if pred_label == "FAKE":
        print("💬 Interpretation: Model detected traits typical of AI-generated imagery — such as smooth texture or inconsistent edges.")
    else:
        print("💬 Interpretation: Model found natural texture, consistent boundaries, and symmetrical features typical of real faces.")
    print("--------------------------------\n")

# ========== CLI ==========
def main():
    print("⚙️ Loading model... (please wait)")
    try:
        model = load_model()
    except Exception as e:
        print("❌ Failed to load model:", e)
        if DEBUG: traceback.print_exc()
        return

    print("✅ Model ready. Type image paths to analyze (or 'exit' to quit).")
    print(f"FAST mode={FAST}, DEBUG={DEBUG}\n")

    while True:
        path = input("📷 Image path: ").strip().replace('"','')
        if path.lower() in ('exit', 'quit'):
            print("👋 Exiting.")
            break
        if not os.path.exists(path):
            print("❌ File not found.\n")
            continue
        try:
            justify_image(model, path)
        except Exception as e:
            print(f"⚠️ Error analyzing {path}: {e}")
            if DEBUG:
                traceback.print_exc()
            print()

if __name__ == "__main__":
    main()
