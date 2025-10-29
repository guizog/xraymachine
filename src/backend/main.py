from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import UnidentifiedImageError
import torch
import torch.nn as nn
import timm
import os
import cv2
import numpy as np
import traceback

# ==========================
# Configurações
# ==========================
MODEL_NAME = "efficientnet_b4"
MODEL_PATH_MALE = os.getenv("BONE_AGE_MODEL_MALE", r"C:\Users\Administrator\Desktop\main-xraymachine\main-xraymachine\src\backend\male_regression.pth")
MODEL_PATH_FEMALE = os.getenv("BONE_AGE_MODEL_FEMALE", r"C:\Users\Administrator\Desktop\main-xraymachine\main-xraymachine\src\backend\female_regression.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================
# Classe do modelo
# ==========================
class BoneAgeModel(nn.Module):
    def __init__(self, backbone_name="efficientnet_b0", pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg", in_chans=1
        )
        feat_dim = self.backbone.num_features
        self.fc1 = nn.Linear(feat_dim, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        feats = self.backbone(x)
        x = self.fc1(feats)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# ==========================
# Pré-processamento da imagem
# ==========================
def preprocess_xray_from_upload(file, size=380, margin_ratio=0.2, aspect_tolerance=0.4):
    """Pré-processa imagem enviada via UploadFile (FastAPI)"""
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Não foi possível abrir a imagem enviada")

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)

    # Máscara
    _, thresh = cv2.threshold(img_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    use_full = False
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        margin_x = int(margin_ratio * w)
        margin_y = int(margin_ratio * h)
        x = max(x - margin_x, 0)
        y = max(y - margin_y, 0)
        w = min(w + 2 * margin_x, img.shape[1] - x)
        h = min(h + 2 * margin_y, img.shape[0] - y)

        crop_aspect = w / h
        img_aspect = img.shape[1] / img.shape[0]
        if abs(crop_aspect - img_aspect) > aspect_tolerance:
            use_full = True
    else:
        use_full = True

    if use_full:
        hand_crop = cv2.bitwise_and(img_clahe, img_clahe, mask=thresh)
    else:
        hand_crop = img_clahe[y:y + h, x:x + w]
        mask_crop = thresh[y:y + h, x:x + w]
        hand_crop = cv2.bitwise_and(hand_crop, hand_crop, mask=mask_crop)

    # Resize + padding
    h_ratio = size / hand_crop.shape[0]
    w_ratio = size / hand_crop.shape[1]
    scale = min(h_ratio, w_ratio)

    new_w, new_h = int(hand_crop.shape[1] * scale), int(hand_crop.shape[0] * scale)
    resized = cv2.resize(hand_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    top = (size - new_h) // 2
    bottom = size - new_h - top
    left = (size - new_w) // 2
    right = size - new_w - left

    final_img = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                   cv2.BORDER_CONSTANT, value=0)

    return final_img


# ==========================
# Carregamento dos modelos
# ==========================
def load_model(path: str):
    try:
        print(f"Tentando carregar modelo: {path}")
        obj = torch.load(path, map_location=DEVICE, weights_only=False)

        if isinstance(obj, dict):
            print("Checkpoint detectado: state_dict")
            model = BoneAgeModel(backbone_name=MODEL_NAME, pretrained=False)
            model.load_state_dict(obj)
        else:
            print("Checkpoint detectado: modelo inteiro")
            model = obj

        model.to(DEVICE)
        model.eval()
        print(f"✅ Modelo carregado de {path}")
        return model

    except Exception as e:
        traceback.print_exc()
        print(f"❌ Erro ao carregar modelo {path}: {e}")
        return None


model_male = load_model(MODEL_PATH_MALE)
model_female = load_model(MODEL_PATH_FEMALE)


# ==========================
# FastAPI App
# ==========================
app = FastAPI(title="Bone Age Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

<<<<<<< Updated upstream
if os.path.exists("xray_model.h5"):
   print("model already exists, using the saved model")
   #modelAi.runAi()#file_Location)
else:
    print("model does not exist, training the model...")
    modelAi.trainModel()
=======
>>>>>>> Stashed changes

# ==========================
# Rota de predição
# ==========================
@app.post("/predict")
async def predict(file: UploadFile, sexo: str = Form(...)):
    print(f"[DEBUG] Valor recebido no form sexo: '{sexo}'")

    # Normaliza entrada
    sexo = sexo.strip().upper()
    print(f"[DEBUG] Após normalização: '{sexo}'")

    # Seleciona modelo conforme sexo
    if sexo == "M":
        if model_male is None:
            raise HTTPException(status_code=500, detail="Modelo masculino não carregado")
        selected_model = model_male
        sexo_legivel = "Masculino"
        print("[DEBUG] Selecionado modelo MASCULINO")

    elif sexo == "F":
        if model_female is None:
            raise HTTPException(status_code=500, detail="Modelo feminino não carregado")
        selected_model = model_female
        sexo_legivel = "Feminino"
        print("[DEBUG] Selecionado modelo FEMININO")

    else:
        print(f"[DEBUG] Valor inesperado: '{sexo}'")
        raise HTTPException(status_code=400, detail=f"Sexo inválido: {sexo}")

    # Verifica se arquivo é imagem
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Arquivo enviado não é uma imagem válida.")

    try:
        # Pré-processar a imagem (grayscale, CLAHE, crop, resize, padding)
        processed_img = preprocess_xray_from_upload(file.file, size=380)

        # Converter para tensor PyTorch (1 canal)
        img_tensor = torch.tensor(processed_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        img_tensor = (img_tensor / 255.0 - 0.5) / 0.5  # normalização [-1, 1]
        img_tensor = img_tensor.to(DEVICE)

<<<<<<< Updated upstream
    image_bytes: bytes = file.file;
    return {
        "message": "success",
        "results": {
            "class": result,
            "file": "a",#Response(content=image_bytes, media_type="image/png"),
            "boneAge": 10
        }
    }
=======
        # Predição
        with torch.no_grad():
            pred = selected_model(img_tensor).item()
>>>>>>> Stashed changes

        print(f"[DEBUG] Predição feita. Retornando sexo: {sexo_legivel}, idade: {round(pred, 1)} meses")

        return JSONResponse({
            "success": True,
            "idade_predita_meses": round(pred, 1),
            "sexo": sexo_legivel
        })

    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Não foi possível abrir a imagem")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")