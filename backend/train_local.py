import os, random, io
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import timm
from huggingface_hub import hf_hub_download, HfApi

# ── Config ────────────────────────────

WILD_DIR  = r"D:\dataset"
HF_TOKEN  = "YOUR_HF_TOKEN_HERE"
HF_REPO   = "YOUR_HF_REPO_HERE"

SKIP_XCEPTION     = True   
SKIP_EFFICIENTNET = True
SKIP_MESONET      = False
EPOCHS = 1

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── JPEG Augmentation ─────────────────
class JPEGCompression:
    def __init__(self, quality_range=(50, 95)):
        self.quality_range = quality_range
    def __call__(self, img):
        quality = random.randint(*self.quality_range)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        return Image.open(buf).convert('RGB')

# ── Dataset ───────────────────────────
class WildDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), label

# ── Transforms ────────────────────────
transform = transforms.Compose([
    JPEGCompression(quality_range=(50, 95)),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.4, contrast=0.4,
        saturation=0.4, hue=0.1),
    transforms.RandomRotation(10),
    transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

xception_transform = transforms.Compose([
    JPEGCompression(quality_range=(50, 95)),
    transforms.Resize((299, 299)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.4, contrast=0.4,
        saturation=0.4, hue=0.1),
    transforms.RandomRotation(10),
    transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])


def finetune_and_upload(model, loader, epochs, lr,
                        name, filename,
                        is_efficientnet=False):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.5)
    best_acc = 0.0
    model.train()
    for epoch in range(epochs):
        total_loss, correct, total = 0, 0, 0
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            if is_efficientnet:
                outputs = model(images).logits
            else:
                outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        scheduler.step()
        acc = correct / total * 100
        print(f"  Epoch {epoch+1}/{epochs}: "
              f"Loss={total_loss:.3f} Acc={acc:.1f}%")
        torch.save(model.state_dict(), filename)
        if acc > best_acc:
            best_acc = acc
            print(f"  New best: {best_acc:.1f}%")
    api = HfApi()
    print(f"Uploading {filename} to HuggingFace...")
    api.upload_file(
        path_or_fileobj=filename,
        path_in_repo=filename,
        repo_id=HF_REPO,
        token=HF_TOKEN)
    print(f"{name} done! Best: {best_acc:.1f}%")
    return model


# ── MUST have this for Windows! ───────
if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU: (cpu)")

    print("Loading dataset...")
    samples = []
    for split in ['train', 'valid']:
        real_dir = os.path.join(WILD_DIR, split, 'real')
        fake_dir = os.path.join(WILD_DIR, split, 'fake')
        for f in os.listdir(real_dir):
            if f.endswith(('.jpg', '.png', '.jpeg')):
                samples.append((os.path.join(real_dir, f), 0))
        for f in os.listdir(fake_dir):
            if f.endswith(('.jpg', '.png', '.jpeg')):
                samples.append((os.path.join(fake_dir, f), 1))

    random.shuffle(samples)
    real_count = sum(1 for _, l in samples if l == 0)
    fake_count = sum(1 for _, l in samples if l == 1)
    print(f"Total: {len(samples)} samples")
    print(f"REAL: {real_count} ({real_count/len(samples)*100:.1f}%)")
    print(f"FAKE: {fake_count} ({fake_count/len(samples)*100:.1f}%)")

    # ── 1. Xception ───────────────────
    if not SKIP_XCEPTION:
        print("\n" + "="*40)
        print("Fine-tuning Xception...")
        print("="*40)
        xception = timm.create_model(
            'legacy_xception', pretrained=False, num_classes=2)
        weights = hf_hub_download(
            repo_id=HF_REPO, filename="xception_dfdc.pth")
        xception.load_state_dict(
            torch.load(weights, map_location='cpu',
                       weights_only=True), strict=False)
        print("Loaded Xception weights!")
        loader_x = DataLoader(
            WildDataset(samples, xception_transform),
            batch_size=16, shuffle=True,
            num_workers=0, pin_memory=True)
        finetune_and_upload(
            xception, loader_x, EPOCHS, 5e-5,
            "Xception", "xception_dfdc.pth")
        del xception, loader_x
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        print("GPU memory cleared!")
    else:
        print("\nSkipping Xception (already trained!)")

   

    # ── 2. EfficientNet ───────────────
    if not SKIP_EFFICIENTNET:
        print("\n" + "="*40)
        print("Fine-tuning EfficientNet...")
        print("="*40)
        # Use timm here (not transformers) to avoid torch>=2.6 restriction
        # triggered by transformers' safe-loading checks for .bin checkpoints.
        efficientnet = timm.create_model(
            "tf_efficientnet_b7_ns", pretrained=True, num_classes=2
        )
        weights = hf_hub_download(
            repo_id=HF_REPO, filename="efficientnet_dfdc.pth"
        )
        efficientnet.load_state_dict(
            torch.load(weights, map_location="cpu", weights_only=True),
            strict=False,
        )
        print("Loaded EfficientNet weights!")
        for param in efficientnet.parameters():
            param.requires_grad = True
        loader_e = DataLoader(
            WildDataset(samples, transform),
            batch_size=4, shuffle=True,
            num_workers=0, pin_memory=True)
        finetune_and_upload(
            efficientnet, loader_e, EPOCHS, 5e-6,
            "EfficientNet", "efficientnet_dfdc.pth")
        del efficientnet, loader_e
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        print("GPU memory cleared!")
    else:
        print("\nSkipping EfficientNet (already trained!)")

    # ── 3. MesoNet ────────────────────
    if not SKIP_MESONET:
        print("\n" + "="*40)
        print("Fine-tuning MesoNet...")
        print("="*40)
        mesonet = timm.create_model(
            'efficientnet_b0', pretrained=False, num_classes=2)
        weights = hf_hub_download(
            repo_id=HF_REPO, filename="mesonet_dfdc.pth")
        mesonet.load_state_dict(
            torch.load(weights, map_location='cpu',
                       weights_only=True))
        print("Loaded MesoNet weights!")
        loader_m = DataLoader(
            WildDataset(samples, transform),
            batch_size=16, shuffle=True,
            num_workers=0, pin_memory=True)
        finetune_and_upload(
            mesonet, loader_m, EPOCHS, 5e-5,
            "MesoNet", "mesonet_dfdc.pth")
    else:
        print("\nSkipping MesoNet (already trained!)")

    print("\nDONE fine-tuning requested models.")
    print(f"You can now delete: {WILD_DIR}")