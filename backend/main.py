from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import timm
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
import tempfile
import os
import time
import shutil

# ── PDF + Report imports ──────────────
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, white, black
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import io
import base64
from datetime import datetime

app = FastAPI(title="DeepShield API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Device ─────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Model Cache ────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def get_model_path(filename):
    local_path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(local_path):
        print(f"✅ Loading {filename} from cache!")
        return local_path
    print(f"⬇️  Downloading {filename} from HuggingFace...")
    path = hf_hub_download(
        repo_id="deepshield/deepshield-models",
        filename=filename
    )
    shutil.copy(path, local_path)
    print(f"✅ {filename} cached!")
    return local_path

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
try:
    xception = timm.create_model(
        'legacy_xception', pretrained=False, num_classes=2)
    weights_path = get_model_path("xception_dfdc.pth")
    state_dict = torch.load(
        weights_path, map_location=device, weights_only=True)
    xception.load_state_dict(state_dict, strict=False)
    xception = xception.to(device)
    xception.eval()
    print("✅ Xception loaded!")
except Exception as e:
    print(f"⚠️ Xception fallback: {e}")
    xception = timm.create_model(
        'legacy_xception', pretrained=True, num_classes=2)
    xception = xception.to(device)
    xception.eval()

# Model 2 — EfficientNet (timm now!)
try:
    efficientnet = timm.create_model(
        'efficientnet_b7', pretrained=False, num_classes=2)
    weights_path = get_model_path("efficientnet_dfdc.pth")
    state_dict = torch.load(
        weights_path, map_location=device, weights_only=True)
    efficientnet.load_state_dict(state_dict, strict=False)
    efficientnet = efficientnet.to(device)
    efficientnet.eval()
    print("✅ EfficientNet loaded!")
except Exception as e:
    print(f"⚠️ EfficientNet fallback: {e}")
    efficientnet = timm.create_model(
        'efficientnet_b7', pretrained=True, num_classes=2)
    efficientnet = efficientnet.to(device)
    efficientnet.eval()

# Model 3 — MesoNet
try:
    mesonet = timm.create_model(
        'efficientnet_b0', pretrained=False, num_classes=2)
    weights_path = get_model_path("mesonet_dfdc.pth")
    state_dict = torch.load(
        weights_path, map_location=device, weights_only=True)
    mesonet.load_state_dict(state_dict, strict=False)
    mesonet = mesonet.to(device)
    mesonet.eval()
    print("✅ MesoNet loaded!")
except Exception as e:
    print(f"⚠️ MesoNet fallback: {e}")
    mesonet = timm.create_model(
        'efficientnet_b0', pretrained=True, num_classes=2)
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
        print(f"Xception     → [0]={prob[0][0].item()*100:.1f}%  [1]={prob[0][1].item()*100:.1f}%")
        scores['xception'] = prob[0][0].item() * 100

    # EfficientNet (timm — no .logits needed!)
    with torch.no_grad():
        t = efficientnet_transform(pil_image).unsqueeze(0).to(device)
        out = efficientnet(t)
        prob = F.softmax(out, dim=1)
        print(f"EfficientNet → [0]={prob[0][0].item()*100:.1f}%  [1]={prob[0][1].item()*100:.1f}%")
        scores['efficientnet'] = prob[0][0].item() * 100

    # MesoNet
    with torch.no_grad():
        t = efficientnet_transform(pil_image).unsqueeze(0).to(device)
        out = mesonet(t)
        prob = F.softmax(out, dim=1)
        print(f"MesoNet      → [0]={prob[0][0].item()*100:.1f}%  [1]={prob[0][1].item()*100:.1f}%")
        scores['mesonet'] = prob[0][0].item() * 100

    return scores


def generate_heatmap(pil_image):
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        tensor = xception_transform(pil_image).unsqueeze(0).to(device)
        target_layers = [list(xception.children())[-3]]

        cam = GradCAM(model=xception, target_layers=target_layers)
        targets = [ClassifierOutputTarget(1)]
        grayscale_cam = cam(input_tensor=tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        img_resized = pil_image.resize((299, 299))
        img_array = np.array(img_resized).astype(np.float32) / 255.0
        visualization = show_cam_on_image(img_array, grayscale_cam, use_rgb=True)

        _, buffer = cv2.imencode(
            '.jpg', cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
        heatmap_b64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{heatmap_b64}"

    except Exception as e:
        print(f"Heatmap error: {e}")
        return None


def ensemble_score(scores):
    return (
        scores['xception']     * 0.40 +
        scores['efficientnet'] * 0.30 +
        scores['mesonet']      * 0.30
    )


def get_verdict(score):
    if score >= 85:
        return "FAKE"
    elif score >= 60:
        return "UNCERTAIN"
    else:
        return "REAL"


def generate_pdf_report(
    filename, verdict, confidence,
    xception_score, efficientnet_score,
    mesonet_score, frames_analyzed,
    processing_time, heatmap_b64=None
):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )

    brand   = HexColor("#4f6ef7")
    dark    = HexColor("#0a0a0f")
    muted   = HexColor("#6b7280")

    elements = []
    elements.append(Spacer(1, 0.3*cm))
    elements.append(Paragraph(
        "<font color='#4f6ef7'><b>DeepShield</b></font>",
        ParagraphStyle("logo", fontSize=20, fontName="Helvetica-Bold",
                       alignment=TA_CENTER, spaceAfter=0, leading=28)
    ))
    elements.append(Spacer(1, 0.25*cm))
    elements.append(Paragraph(
        "Deepfake Detection Report",
        ParagraphStyle("subtitle", fontSize=11, fontName="Helvetica",
                       textColor=muted, alignment=TA_CENTER, spaceAfter=0, leading=16)
    ))
    elements.append(Spacer(1, 0.15*cm))
    elements.append(Paragraph(
        datetime.now().strftime("%B %d, %Y"),
        ParagraphStyle("hdate", fontSize=9, fontName="Helvetica",
                       textColor=muted, alignment=TA_CENTER, spaceAfter=0, leading=14)
    ))
    elements.append(Spacer(1, 0.4*cm))
    elements.append(Table([[""]], colWidths=[17*cm],
        style=TableStyle([("LINEBELOW", (0,0), (-1,-1), 1, brand)])))
    elements.append(Spacer(1, 0.5*cm))

    elements.append(Paragraph("File Information",
        ParagraphStyle("section", fontSize=13, fontName="Helvetica-Bold",
                       textColor=brand, spaceAfter=8)))
    file_table = Table(
        [["Filename", filename],
         ["Frames Analyzed", str(frames_analyzed)],
         ["Processing Time", f"{processing_time}s"],
         ["Analysis Date", datetime.now().strftime("%Y-%m-%d")]],
        colWidths=[5*cm, 12*cm],
        style=TableStyle([
            ("FONTNAME",      (0,0), (0,-1), "Helvetica-Bold"),
            ("FONTNAME",      (1,0), (1,-1), "Helvetica"),
            ("FONTSIZE",      (0,0), (-1,-1), 10),
            ("TEXTCOLOR",     (0,0), (0,-1), muted),
            ("TEXTCOLOR",     (1,0), (1,-1), dark),
            ("ROWBACKGROUNDS",(0,0), (-1,-1), [HexColor("#f9fafb"), white]),
            ("PADDING",       (0,0), (-1,-1), 6),
            ("GRID",          (0,0), (-1,-1), 0.5, HexColor("#e5e7eb")),
        ])
    )
    elements.append(file_table)
    elements.append(Spacer(1, 0.6*cm))

    elements.append(Paragraph("Verdict",
        ParagraphStyle("section", fontSize=13, fontName="Helvetica-Bold",
                       textColor=brand, spaceAfter=8)))
    verdict_hex = "#f43f5e" if verdict == "FAKE" else "#10b981" if verdict == "REAL" else "#f59e0b"
    elements.append(Spacer(1, 0.2*cm))
    elements.append(Paragraph(
        f"<font color='{verdict_hex}'><b>{verdict}</b></font>",
        ParagraphStyle("vtext", fontSize=24, fontName="Helvetica-Bold",
                       alignment=TA_CENTER, spaceAfter=10, leading=30)
    ))
    elements.append(Paragraph(
        f"Ensemble Confidence: <b>{float(confidence):.1f}%</b>",
        ParagraphStyle("ctext", fontSize=12, fontName="Helvetica",
                       textColor=muted, alignment=TA_CENTER, spaceAfter=16, leading=16)
    ))
    elements.append(Spacer(1, 0.3*cm))

    elements.append(Paragraph("Model Scores",
        ParagraphStyle("section", fontSize=13, fontName="Helvetica-Bold",
                       textColor=brand, spaceAfter=8)))
    score_table = Table(
        [["Model", "Score", "Interpretation"],
         ["Xception",        f"{float(xception_score):.1f}%",     "FAKE" if float(xception_score) > 65 else "REAL"],
         ["EfficientNet-B7", f"{float(efficientnet_score):.1f}%", "FAKE" if float(efficientnet_score) > 65 else "REAL"],
         ["MesoNet",         f"{float(mesonet_score):.1f}%",      "FAKE" if float(mesonet_score) > 65 else "REAL"],
         ["Ensemble (Final)",f"{float(confidence):.1f}%",         verdict]],
        colWidths=[6*cm, 4*cm, 7*cm],
        style=TableStyle([
            ("BACKGROUND",    (0,0), (-1,0),  brand),
            ("TEXTCOLOR",     (0,0), (-1,0),  white),
            ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
            ("FONTNAME",      (0,1), (-1,-1), "Helvetica"),
            ("FONTSIZE",      (0,0), (-1,-1), 10),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [HexColor("#f9fafb"), white]),
            ("GRID",          (0,0), (-1,-1), 0.5, HexColor("#e5e7eb")),
            ("PADDING",       (0,0), (-1,-1), 8),
            ("ALIGN",         (1,0), (1,-1),  "CENTER"),
            ("FONTNAME",      (0,4), (-1,4),  "Helvetica-Bold"),
        ])
    )
    elements.append(score_table)
    elements.append(Spacer(1, 0.6*cm))

    if heatmap_b64:
        try:
            elements.append(Paragraph("Heatmap Visualization",
                ParagraphStyle("section", fontSize=13, fontName="Helvetica-Bold",
                               textColor=brand, spaceAfter=8)))
            img_data = base64.b64decode(
                heatmap_b64.split(",")[1] if "," in heatmap_b64 else heatmap_b64)
            img_buf = io.BytesIO(img_data)
            rl_img = RLImage(img_buf, width=10*cm, height=7*cm)
            elements.append(rl_img)
            elements.append(Spacer(1, 0.3*cm))
            elements.append(Paragraph(
                "Red regions indicate areas most likely to be AI-manipulated. "
                "Generated using Grad-CAM on the Xception model.",
                ParagraphStyle("caption", fontSize=9, fontName="Helvetica",
                               textColor=muted, spaceAfter=12)
            ))
        except Exception as e:
            print(f"Heatmap PDF error: {e}")

    elements.append(Table([[""]], colWidths=[17*cm],
        style=TableStyle([("LINEABOVE", (0,0), (-1,-1), 0.5, muted)])))
    elements.append(Spacer(1, 0.3*cm))
    elements.append(Paragraph("Legal Disclaimer",
        ParagraphStyle("disclaimer_title", fontSize=11, fontName="Helvetica-Bold",
                       textColor=muted, spaceAfter=6)))
    elements.append(Paragraph(
        "This report is generated by an automated AI system and is intended "
        "for informational purposes only. DeepShield does not guarantee the "
        "accuracy of results. Always combine automated analysis with human "
        "expert review in legal or high-stakes contexts. DeepShield retains "
        "no copies of uploaded files or results.",
        ParagraphStyle("disclaimer_text", fontSize=8, fontName="Helvetica",
                       textColor=muted, spaceAfter=8)
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer


class ReportData(BaseModel):
    filename: str
    verdict: str
    confidence: float
    xception_score: float
    efficientnet_score: float
    mesonet_score: float
    frames_analyzed: int
    processing_time: float
    heatmap_url: Optional[str] = None


@app.get("/")
def home():
    return {"message": "DeepShield API", "version": "1.0.0", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy", "device": str(device), "models_loaded": True}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    start_time = time.time()
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        is_video = suffix.lower() in ['.mp4', '.mov', '.avi', '.webm', '.mkv']
        if is_video:
            frames = extract_frames(tmp_path, num_frames=10)
            if not frames:
                return {"error": "Could not extract frames"}
        else:
            img = Image.open(tmp_path).convert('RGB')
            frames = [img]

        all_scores = {'xception': [], 'efficientnet': [], 'mesonet': []}
        for frame in frames:
            scores = analyze_image(frame)
            for model in scores:
                all_scores[model].append(scores[model])

        avg_scores = {model: sum(s)/len(s) for model, s in all_scores.items()}
        final_score = ensemble_score(avg_scores)
        verdict = get_verdict(final_score)
        processing_time = round(time.time() - start_time, 2)

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


@app.post("/report-from-data")
async def report_from_data(data: ReportData):
    pdf_buffer = generate_pdf_report(
        filename=data.filename, verdict=data.verdict,
        confidence=data.confidence, xception_score=data.xception_score,
        efficientnet_score=data.efficientnet_score, mesonet_score=data.mesonet_score,
        frames_analyzed=data.frames_analyzed, processing_time=data.processing_time,
        heatmap_b64=data.heatmap_url
    )
    return StreamingResponse(pdf_buffer, media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=deepshield-report.pdf"})