import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()

    # =============================
    # CONFIGURATION
    # =============================
    train_dir = r"D:\celeb A\stylegan3\dataset\train"
    test_dir = r"D:\celeb A\stylegan3\dataset\test"
    checkpoint_dir = "checkpoints_v3_fixed"
    os.makedirs(checkpoint_dir, exist_ok=True)

    batch_size = 16
    epochs = 15
    lr = 1e-4
    model_name = "swin_base_patch4_window7_224"

    resume_checkpoint = True
    checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint_v3.pth")

    # =============================
    # DEVICE SETUP
    # =============================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Using device: {device}")

    # =============================
    # DATA TRANSFORMS (Heavy Augmentation)
    # =============================
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.GaussianBlur(3),
        transforms.RandomAdjustSharpness(2),
        transforms.RandomAutocontrast(),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    # =============================
    # DATA LOADERS
    # =============================
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # =============================
    # MODEL INITIALIZATION
    # =============================
    model = timm.create_model(model_name, pretrained=True)

    # Replace classifier for binary classification
    if hasattr(model, 'head') and isinstance(model.head, nn.Linear):
        model.head = nn.Linear(model.head.in_features, 2)
    elif hasattr(model, 'head') and hasattr(model.head, 'fc'):
        model.head.fc = nn.Linear(model.head.fc.in_features, 2)
    elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
        model.classifier = nn.Linear(model.classifier.in_features, 2)
    else:
        raise ValueError("❌ Couldn't find classification head in this Swin model structure.")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    start_epoch = 0

    # =============================
    # LOAD CHECKPOINT
    # =============================
    if resume_checkpoint and os.path.exists(checkpoint_path):
        print(f"🔄 Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"✅ Resumed from epoch {start_epoch}")

    # =============================
    # TRAINING LOOP
    # =============================
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss, correct = 0.0, 0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        train_acc = 100 * correct / len(train_dataset)
        avg_loss = total_loss / len(train_loader)

        # =============================
        # VALIDATION
        # =============================
        model.eval()
        correct_test = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(imgs)
                correct_test += (outputs.argmax(1) == labels).sum().item()

        test_acc = 100 * correct_test / len(test_dataset)
        print(f"📊 Epoch {epoch+1}: Loss={avg_loss:.4f} | Train Acc={train_acc:.2f}% | Test Acc={test_acc:.2f}%")

        # =============================
        # SAVE CHECKPOINTS
        # =============================
        epoch_ckpt = os.path.join(checkpoint_dir, f"epoch_{epoch+1}_swinv3.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_acc": train_acc,
            "test_acc": test_acc
        }, epoch_ckpt)

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }, checkpoint_path)

        print(f"💾 Checkpoint saved: {epoch_ckpt}")

    print("✅ Training complete! Final model saved in:", checkpoint_dir)
