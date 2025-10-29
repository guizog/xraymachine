# =========================================================
# maincopy.py - FastAPI + BoneAgeModel na CPU (EfficientNet-B3, 20 classes)
# Retorna idade contínua em anos a partir de modelo de classificação
# =========================================================

from fastapi import FastAPI, File, UploadFile
from typing import Dict
from PIL import Image
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import timm

# =========================================================
# 1️⃣ Definição do modelo (20 classes)
# =========================================================
class BoneAgeModel(nn.Module):
    def __init__(self, backbone_name='efficientnet_b3', pretrained=True, num_classes=20):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained,
                                         num_classes=0, global_pool='avg')
        # fc1 recebe +1 feature (sexo)
        self.fc1 = nn.Linear(self.backbone.num_features + 1, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)  # 20 classes

    def forward(self, x, sex):
        features = self.backbone(x)
        features = torch.cat([features, sex], dim=1)
        x = self.fc1(features)
        x = self.relu(x)
        x = self.fc2(x)  # logits das 20 classes
        return x

# =========================================================
# 2️⃣ Carregar modelo e pesos
# =========================================================
device = torch.device("cpu")
model = BoneAgeModel(backbone_name='efficientnet_b3', pretrained=False, num_classes=20)
state_dict = torch.load("finalv1.pth", map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# =========================================================
# 3️⃣ Função para processar idade contínua
# =========================================================
def predict_age_continuous(model, image: Image.Image, male=False) -> float:
    device = next(model.parameters()).device

    # Transformações da imagem
    test_tfms = A.Compose([
        A.Resize(300, 300),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    img_tensor = test_tfms(image=np.array(image))['image'].unsqueeze(0).to(device)
    sex_tensor = torch.tensor([[1.0 if male else 0.0]], device=device)

    with torch.no_grad():
        output = model(img_tensor, sex_tensor)          # logits das 20 classes
        probs = torch.softmax(output, dim=1)           # probabilidades
        ages = torch.arange(probs.shape[1], dtype=torch.float32)  # 0..19 anos
        predicted_age = torch.sum(probs * ages)        # idade contínua (esperança ponderada)
    
    return predicted_age.item()

# =========================================================
# 4️⃣ API FastAPI
# =========================================================
app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile, male: bool = False) -> Dict:
    try:
        image = Image.open(file.file).convert("RGB")
        age = predict_age_continuous(model, image, male)
        return {
            "predicted_age": age
        }
    except Exception as e:
        return {"error": str(e)}
