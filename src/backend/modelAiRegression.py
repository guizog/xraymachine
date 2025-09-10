import os, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_device():
    if torch.cuda.is_available():
        print(">> Usando CUDA (NVIDIA)")
        return torch.device("cuda")
    try:
        if getattr(torch, 'version', None) and getattr(torch.version, 'hip', None):
            print(">> Usando ROCm (AMD Linux)")
            return torch.device("cuda")
    except Exception:
        pass
    try:
        import torch_directml
        print(">> Usando DirectML (AMD/Intel no Windows)")
        return torch_directml.device()
    except Exception:
        pass
    print(">> Usando CPU")
    return torch.device("cpu")

device = get_device()


IMG_DIR = r"images"  # pasta com imagens .png
CSV_PATH = r"labels.csv"

IMG_SIZE = 224
BACKBONES = {224: "efficientnet_b0", 300: "efficientnet_b3", 380: "efficientnet_b4", 512: "efficientnet_b6"}
BACKBONE = BACKBONES.get(IMG_SIZE, "efficientnet_b0")
print(f"Usando backbone {BACKBONE} com resolução {IMG_SIZE}x{IMG_SIZE}")


class BoneAgeDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['id']}.png")
        image = np.array(Image.open(img_path).convert('RGB'))
        label = row['boneage'] / 12.0   # idade em anos
        sex = np.float32(1.0 if row['male'] else 0.0)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, sex, torch.tensor(label, dtype=torch.float32)


train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


def build_model_regression():
    base_model = getattr(models, BACKBONE)(weights="DEFAULT")
    feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
    feature_dim = base_model.classifier[1].in_features
    regressor = nn.Linear(feature_dim + 1, 1)
    return feature_extractor, regressor


def train_model_regression(df, img_dir, epochs=10, batch_size=16, lr=1e-4, patience=5):
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    train_dataset = BoneAgeDataset(train_df, img_dir, transform=train_transform)
    val_dataset   = BoneAgeDataset(val_df, img_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    feature_extractor, regressor = build_model_regression()
    feature_extractor, regressor = feature_extractor.to(device), regressor.to(device)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(list(feature_extractor.parameters()) + list(regressor.parameters()), lr=lr)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        feature_extractor.train(); regressor.train()
        train_loss = 0.0
        for images, sex, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            images, sex, labels = images.to(device), sex.to(device).unsqueeze(1).float(), labels.to(device)
            optimizer.zero_grad()
            features = feature_extractor(images).view(images.size(0), -1)
            features = torch.cat([features, sex], dim=1)
            outputs = regressor(features).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_loader.dataset)

        feature_extractor.eval(); regressor.eval()
        val_loss, y_true, y_pred = 0.0, [], []
        with torch.no_grad():
            for images, sex, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                images, sex, labels = images.to(device), sex.to(device).unsqueeze(1).float(), labels.to(device)
                features = feature_extractor(images).view(images.size(0), -1)
                features = torch.cat([features, sex], dim=1)
                outputs = regressor(features).squeeze(1)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                y_pred.extend(outputs.cpu().numpy())
                y_true.extend(labels.cpu().numpy())
        val_loss /= len(val_loader.dataset)

        mae = mean_absolute_error(y_true, y_pred)
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_MAE={mae:.2f} anos")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'feature_extractor': feature_extractor.state_dict(),
                'regressor': regressor.state_dict(),
                'img_size': IMG_SIZE,
                'backbone': BACKBONE
            }, "best_model_regression.pth")
            print(">> Modelo salvo com menor val_loss")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f">> Patience {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping ativado.")
                break

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return feature_extractor, regressor


def evaluate_model_regression(model_path, test_df, img_dir):
    checkpoint = torch.load(model_path, map_location="cpu")
    img_size = checkpoint.get('img_size', 224)
    backbone = checkpoint.get('backbone', "efficientnet_b0")

    base_model = getattr(models, backbone)(weights="DEFAULT")
    feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
    feature_dim = base_model.classifier[1].in_features
    regressor = nn.Linear(feature_dim + 1, 1)

    feature_extractor.load_state_dict(checkpoint['feature_extractor'])
    regressor.load_state_dict(checkpoint['regressor'])
    feature_extractor, regressor = feature_extractor.to(device), regressor.to(device)

    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    feature_extractor.eval(); regressor.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Testing"):
            img_path = os.path.join(img_dir, f"{row['id']}.png")
            sex_tensor = torch.tensor([[1.0 if row['male'] else 0.0]], dtype=torch.float32).to(device)
            image = np.array(Image.open(img_path).convert("RGB"))
            image = transform(image=image)["image"].unsqueeze(0).to(device)
            features = feature_extractor(image).view(1, -1)
            features = torch.cat([features, sex_tensor], dim=1)
            output = regressor(features).item()
            y_pred.append(output)
            y_true.append(row['boneage'] / 12.0)

    mae = mean_absolute_error(y_true, y_pred)
    print(f"MAE: {mae:.2f} anos")

    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel("Idade Real (anos)")
    plt.ylabel("Idade Prevista (anos)")
    plt.title("Predição de Idade Óssea - Regressão")
    plt.show()

    return y_true, y_pred


# Exemplo de treino
df = pd.read_csv(CSV_PATH)
print(df.head())

feature_extractor, regressor = train_model_regression(
    df,
    IMG_DIR,
    epochs=10,
    batch_size=16,
    lr=1e-4,
    patience=2
)

# Exemplo de avaliação
TEST_CSV = r"ValidationDataset.csv"
TEST_IMG_DIR = r"boneage-validation"

test_df = pd.read_csv(TEST_CSV)
y_true, y_pred = evaluate_model_regression("best_model_regression.pth", test_df, TEST_IMG_DIR)
