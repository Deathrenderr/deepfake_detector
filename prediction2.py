import torch
import timm
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image

# ====== Configuration ======
checkpoint_path = r"checkpoints_v3_fixed\epoch_15_swinv3.pth"
model_name = "swin_base_patch4_window7_224"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== Load Model ======
model = timm.create_model(model_name, pretrained=False)

# Replace classifier
if hasattr(model, 'head') and isinstance(model.head, nn.Linear):
    model.head = nn.Linear(model.head.in_features, 2)
elif hasattr(model, 'head') and hasattr(model.head, 'fc'):
    model.head.fc = nn.Linear(model.head.fc.in_features, 2)
elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
    model.classifier = nn.Linear(model.classifier.in_features, 2)
else:
    raise ValueError("❌ Couldn't find classification head in this Swin model structure.")

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

# ====== Transform ======
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

label_map = {0: "FAKE", 1: "REAL"}

# ====== Video DeepFake Detector ======
def analyze_video(video_path, frame_rate_sample=1):
    """
    video_path: path to the video file
    frame_rate_sample: number of frames to sample per second
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Could not open video:", video_path)
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames_to_skip = max(1, int(fps / frame_rate_sample))

    frame_count = 0
    real_confidences = []
    fake_confidences = []

    print(f"\n🎥 Analyzing '{video_path}' ({fps} FPS, sampling every {frames_to_skip} frames)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frames_to_skip == 0:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                fake_confidences.append(probs[0].item() * 100)
                real_confidences.append(probs[1].item() * 100)

        frame_count += 1

    cap.release()

    if len(real_confidences) == 0:
        print("⚠️ No frames analyzed.")
        return

    avg_real = np.mean(real_confidences)
    avg_fake = np.mean(fake_confidences)

    print(f"\n📊 Average Confidence:")
    print(f"   REAL: {avg_real:.2f}%")
    print(f"   FAKE: {avg_fake:.2f}%")

    final_label = "REAL" if avg_real > avg_fake else "FAKE"
    print(f"\n🧾 Final Verdict: {final_label}")

# ====== Run Detector ======
if __name__ == "__main__":
    print("\n🧠 DeepFake Video Detector Ready!")
    print("💡 Enter a video path to analyze (type 'exit' to quit)\n")

    while True:
        video_path = input("🎞️ Enter video path: ").strip().replace('"', '')
        if video_path.lower() == "exit":
            print("👋 Exiting video detector. Goodbye!")
            break

        analyze_video(video_path, frame_rate_sample=1)  # 1 frame/sec (adjust if needed)
