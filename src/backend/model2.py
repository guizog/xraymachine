import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from PIL import Image

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ======================================================================================
# 1. CONFIGURAÇÕES CENTRALIZADAS
# ======================================================================================

def get_device():
    """Detecta e retorna o melhor dispositivo disponível para PyTorch."""
    if torch.cuda.is_available():
        print(">> Usando CUDA (NVIDIA)")
        return torch.device("cuda")
    # Adicione outras verificações de dispositivo se necessário (ROCm, DirectML, etc.)
    print(">> Usando CPU")
    return torch.device("cpu")


CONFIG = {
    "img_dir_train": r"images",
    "csv_path_train": r"labels.csv",
    "img_dir_test": r"boneage-validation",
    "csv_path_test": r"ValidationDataset.csv",
    "model_save_path": "best_model_regression.pth",
    "img_size": 224,
    "backbone": "efficientnet_b0",  # 'efficientnet_b0' para 224, 'efficientnet_b3' para 300, etc.
    "batch_size": 16,
    "epochs": 10,
    "lr": 1e-4,
    "patience": 3,
    "device": get_device(),
    "test_split_size": 0.2,
    "random_state": 42,
}


# ======================================================================================
# 2. DATASET E TRANSFORMAÇÕES
# ======================================================================================

class BoneAgeDataset(Dataset):
    """Dataset customizado para o problema de idade óssea."""

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

        # Converte a idade em meses para anos
        label = torch.tensor(row['boneage'] / 12.0, dtype=torch.float32)
        sex = torch.tensor(1.0 if row['male'] else 0.0, dtype=torch.float32)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, sex, label


def get_transforms(img_size):
    """Retorna as transformações de aumento e normalização para treino e validação."""
    common_normalize = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        common_normalize,
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        common_normalize,
        ToTensorV2(),
    ])
    return train_transform, val_transform


# ======================================================================================
# 3. CLASSE DO MODELO
# ======================================================================================

class BoneAgeModel(nn.Module):
    """Encapsula a arquitetura do modelo: extrator de features + regressor."""

    def __init__(self, backbone_name="efficientnet_b0"):
        super().__init__()
        base_model = getattr(models, backbone_name)(weights="DEFAULT")

        # Extrator de features (todas as camadas exceto a última de classificação)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])

        # Dimensão da saída do extrator
        feature_dim = base_model.classifier[1].in_features

        # Regressor que recebe as features da imagem + 1 feature para o sexo
        self.regressor = nn.Linear(feature_dim + 1, 1)

    def forward(self, image, sex):
        # image.shape: (batch, channels, height, width)
        # sex.shape: (batch, 1)
        features = self.feature_extractor(image)
        features = features.view(features.size(0), -1)  # Achatando para (batch, feature_dim)

        # Concatena as features da imagem com a informação de sexo
        combined_features = torch.cat([features, sex.unsqueeze(1)], dim=1)

        output = self.regressor(combined_features)
        return output.squeeze(1)  # Remove a última dimensão para ter shape (batch)


# ======================================================================================
# 4. CLASSE DE TREINAMENTO
# ======================================================================================

class Trainer:
    """Gerencia o ciclo de treinamento e validação do modelo."""

    def __init__(self, model, train_loader, val_loader, config):
        self.model = model.to(config["device"])
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.criterion = nn.L1Loss()  # MAE Loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=config["lr"])
        self.device = config["device"]

    def _train_epoch(self):
        """Executa uma época de treinamento."""
        self.model.train()
        total_loss = 0.0
        for images, sex, labels in tqdm(self.train_loader, desc="Training"):
            images, sex, labels = images.to(self.device), sex.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images, sex)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
        return total_loss / len(self.train_loader.dataset)

    def _validate_epoch(self):
        """Executa uma época de validação."""
        self.model.eval()
        total_loss = 0.0
        y_true, y_pred = [], []
        with torch.no_grad():
            for images, sex, labels in tqdm(self.val_loader, desc="Validating"):
                images, sex, labels = images.to(self.device), sex.to(self.device), labels.to(self.device)

                outputs = self.model(images, sex)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                y_pred.extend(outputs.cpu().numpy())
                y_true.extend(labels.cpu().numpy())

        val_loss = total_loss / len(self.val_loader.dataset)
        val_mae = mean_absolute_error(y_true, y_pred)
        return val_loss, val_mae

    def fit(self):
        """Inicia o loop de treinamento completo com early stopping."""
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.config["epochs"]):
            print(f"\n--- Epoch {epoch + 1}/{self.config['epochs']} ---")
            train_loss = self._train_epoch()
            val_loss, val_mae = self._validate_epoch()

            print(f"Epoch Result: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_MAE={val_mae:.2f} anos")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.config["model_save_path"])
                print(f">> Modelo salvo com val_loss: {best_val_loss:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f">> Patience {patience_counter}/{self.config['patience']}")
                if patience_counter >= self.config['patience']:
                    print("Early stopping ativado. Finalizando o treinamento.")
                    break

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ======================================================================================
# 5. FUNÇÕES DE AVALIAÇÃO E PLOT
# ======================================================================================

def evaluate(model, test_loader, device):
    """Avalia o modelo no conjunto de teste."""
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, sex, labels in tqdm(test_loader, desc="Testing"):
            images, sex = images.to(device), sex.to(device)
            outputs = model(images, sex)
            y_pred.extend(outputs.cpu().numpy())
            y_true.extend(labels.numpy())

    mae = mean_absolute_error(y_true, y_pred)
    print(f"\nResultado da Avaliação - MAE Final: {mae:.2f} anos")
    return y_true, y_pred


def plot_results(y_true, y_pred):
    """Plota os resultados da predição vs. valores reais."""
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5, label="Predições")
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label="Linha Ideal (y=x)")
    plt.xlabel("Idade Real (anos)")
    plt.ylabel("Idade Prevista (anos)")
    plt.title("Predição de Idade Óssea vs. Real")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()


# ======================================================================================
# 6. BLOCO DE EXECUÇÃO PRINCIPAL
# ======================================================================================

if __name__ == "__main__":
    # --- Preparação dos Dados de Treino ---
    df = pd.read_csv(CONFIG["csv_path_train"])
    train_df, val_df = train_test_split(
        df,
        test_size=CONFIG["test_split_size"],
        random_state=CONFIG["random_state"]
    )

    train_transform, val_transform = get_transforms(CONFIG["img_size"])

    train_dataset = BoneAgeDataset(train_df, CONFIG["img_dir_train"], transform=train_transform)
    val_dataset = BoneAgeDataset(val_df, CONFIG["img_dir_train"], transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    # --- Treinamento ---
    print("Iniciando o treinamento...")
    model_train = BoneAgeModel(backbone_name=CONFIG["backbone"])
    trainer = Trainer(model_train, train_loader, val_loader, CONFIG)
    trainer.fit()

    # --- Avaliação ---
    print("\nIniciando a avaliação no conjunto de teste...")
    test_df = pd.read_csv(CONFIG["csv_path_test"])

    # Usamos o val_transform pois não queremos data augmentation na avaliação
    test_dataset = BoneAgeDataset(test_df, CONFIG["img_dir_test"], transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    # Carrega o melhor modelo salvo durante o treino
    model_eval = BoneAgeModel(backbone_name=CONFIG["backbone"])
    model_eval.load_state_dict(torch.load(CONFIG["model_save_path"], map_location=CONFIG["device"]))
    model_eval.to(CONFIG["device"])

    y_true, y_pred = evaluate(model_eval, test_loader, CONFIG["device"])
    plot_results(y_true, y_pred)