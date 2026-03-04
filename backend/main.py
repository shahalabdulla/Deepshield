from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import timm
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
from transformers import AutoModelForImageClassification
from huggingface_hub import hf_hub_download
import tempfile
import os
import time

app = FastAPI(title="DeepShield API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Device ─────────────────────────────
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Device: {device}")

# ── Transforms ─────────────────────────
xception_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

efficientnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ── Load Models ─────────────────────────
print("Loading models...")

# Model 1 — Xception
xception = timm.create_model(
    'xception41', pretrained=False, num_classes=2
)
try:
    weights_path = hf_hub_download(
        repo_id="deepshield/deepshield-models",
        filename="xception_dfdc.pth"
    )
    xception.load_state_dict(
        torch.load(weights_path, map_location=device)
    )
    print("✅ Xception loaded from HuggingFace!")
except:
    print("⚠️ Using base Xception weights")
xception = xception.to(device)
xception.eval()

# Model 2 — EfficientNet
efficientnet = AutoModelForImageClassification.from_pretrained(
    "google/efficientnet-b7",
    num_labels=2,
    ignore_mismatched_sizes=True
)
try:
    weights_path = hf_hub_download(
        repo_id="deepshield/deepshield-models",
        filename="efficientnet_dfdc.pth"
    )
    efficientnet.load_state_dict(
        torch.load(weights_path, map_location=device)
    )
    print("✅ EfficientNet loaded from HuggingFace!")
except:
    print("⚠️ Using base EfficientNet weights")
efficientnet = efficientnet.to(device)
efficientnet.eval()

# Model 3 — MesoNet
mesonet = timm.create_model(
    'efficientnet_b0', pretrained=False, num_classes=2
)
try:
    weights_path = hf_hub_download(
        repo_id="deepshield/deepshield-models",
        filename="mesonet_dfdc.pth"
    )
    mesonet.load_state_dict(
        torch.load(weights_path, map_location=device)
    )
    print("✅ MesoNet loaded from HuggingFace!")
except:
    print("⚠️ Using base MesoNet weights")
mesonet = mesonet.to(device)
mesonet.eval()

print("✅ All models ready!")

# ── Helper Functions ────────────────────
def extract_frames(video_path, num_frames=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return frames
    indices = np.linspace(0, total-1, num_frames, dtype=int)
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
    cap.release()
    return frames

def analyze_image(pil_image):
    scores = {}

    # Xception
    with torch.no_grad():
        t = xception_transform(pil_image).unsqueeze(0).to(device)
        out = xception(t)
        prob = F.softmax(out, dim=1)
        scores['xception'] = prob[0][1].item() * 100

    # EfficientNet
    with torch.no_grad():
        t = efficientnet_transform(pil_image).unsqueeze(0).to(device)
        out = efficientnet(t).logits
        prob = F.softmax(out, dim=1)
        scores['efficientnet'] = prob[0][1].item() * 100

    # MesoNet
    with torch.no_grad():
        t = efficientnet_transform(pil_image).unsqueeze(0).to(device)
        out = mesonet(t)
        prob = F.softmax(out, dim=1)
        scores['mesonet'] = prob[0][1].item() * 100

    return scores


def generate_heatmap(pil_image):
    """Generate Grad-CAM heatmap using Xception"""
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        import cv2
        import base64

        # Prepare image tensor
        tensor = xception_transform(pil_image).unsqueeze(0).to(device)

        # Get last conv layer for Grad-CAM
        target_layers = [xception.blocks[-1]]

        # Generate CAM
        cam = GradCAM(
            model=xception,
            target_layers=target_layers
        )

        # Target class 1 = FAKE
        targets = [ClassifierOutputTarget(1)]
        grayscale_cam = cam(
            input_tensor=tensor,
            targets=targets
        )
        grayscale_cam = grayscale_cam[0, :]

        # Overlay on original image
        img_resized = pil_image.resize((299, 299))
        img_array = np.array(img_resized).astype(np.float32) / 255.0

        visualization = show_cam_on_image(
            img_array,
            grayscale_cam,
            use_rgb=True
        )

        # Convert to base64 for frontend
        _, buffer = cv2.imencode(
            '.jpg',
            cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
        )
        heatmap_b64 = base64.b64encode(
            buffer
        ).decode('utf-8')

        return f"data:image/jpeg;base64,{heatmap_b64}"

    except Exception as e:
        print(f"Heatmap error: {e}")
        return None

def ensemble_score(scores):
    # EfficientNet excluded until retrained
    # Xception 55%, MesoNet 45%
    return (
        scores['xception'] * 0.55 +
        scores['mesonet'] * 0.45
    )

def get_verdict(score):
    if score >= 65:
        return "FAKE"
    elif score >= 40:
        return "UNCERTAIN"
    else:
        return "REAL"

# ── Routes ──────────────────────────────
@app.get("/")
def home():
    return {
        "message": "DeepShield API",
        "version": "1.0.0",
        "status": "running",
        "models": ["xception", "efficientnet", "mesonet"]
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "device": str(device),
        "models_loaded": True
    }

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    start_time = time.time()

    # Save uploaded file temporarily
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=suffix
    ) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        is_video = suffix.lower() in [
            '.mp4', '.mov', '.avi', '.webm', '.mkv'
        ]

        if is_video:
            frames = extract_frames(tmp_path, num_frames=10)
            if not frames:
                return {"error": "Could not extract frames"}
        else:
            img = Image.open(tmp_path).convert('RGB')
            frames = [img]

        # Analyze all frames
        all_scores = {
            'xception': [],
            'efficientnet': [],
            'mesonet': []
        }

        for frame in frames:
            scores = analyze_image(frame)
            for model in scores:
                all_scores[model].append(scores[model])

        # Average across frames
        avg_scores = {
            model: sum(scores) / len(scores)
            for model, scores in all_scores.items()
        }

       # Final verdict
        final_score = ensemble_score(avg_scores)
        verdict = get_verdict(final_score)
        processing_time = round(time.time() - start_time, 2)

        # Generate heatmap on first frame
        print("Generating heatmap...")
        heatmap_url = generate_heatmap(frames[0])
        print("Heatmap done!" if heatmap_url else "Heatmap failed")

        return {
            "verdict": verdict,
            "confidence": round(final_score, 1),
            "xception_score": round(avg_scores['xception'], 1),
            "efficientnet_score": round(avg_scores['efficientnet'], 1),
            "mesonet_score": round(avg_scores['mesonet'], 1),
            "frames_analyzed": len(frames),
            "processing_time": processing_time,
            "heatmap_url": heatmap_url
        }

    finally:
        os.unlink(tmp_path)

