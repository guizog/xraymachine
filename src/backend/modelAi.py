import os
import zipfile
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def trainModel():
    #for pure windows
    dataPath = r"C:\Users\guilherme.zografos\Downloads\Bone+Age+Training+Set+Annotations\train.csv"
    image_dir = r"C:\Users\guilherme.zografos\Documents\Bone+Age+Training+Set\boneage-training-dataset"

    #for wsl build
    #dataPath = "/mnt/c/Users/guizo/Downloads/Bone+Age+Training+Set+Annotations/train.csv"
    #image_dir = "/mnt/c/Users/guizo/Downloads/Bone+Age+Training+Set/boneage-training-dataset"
    csvData = pd.read_csv(dataPath)

    csvData["boneage_class"] = (csvData["boneage"] // 12).clip(0, 18).astype(str)
    csvData["boneage"] = csvData["boneage"].astype(str)
    csvData["id"] = csvData["id"].astype(str) + ".png"

    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(csvData, test_size=0.2, random_state=42)

    datagen = ImageDataGenerator(rescale=1. / 255)
    #all_classes = csvData["boneage"].astype(str).unique().tolist()

    train_generator = datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=image_dir,
        x_col="id",
        y_col="boneage_class",
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical"
        #classes=all_classes,
        #shuffle=True
    )

    val_generator = datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=image_dir,
        x_col="id",
        y_col="boneage_class",
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical"
        #classes=all_classes,
        #shuffle=False
    )

    print("Imagens no treino:", train_generator.samples)
    print("Imagens na validação:", val_generator.samples)

    num_classes = len(train_generator.class_indices)
    print("Número de classes detectadas:", num_classes)

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dense(num_classes, activation="softmax")  # multi-classe
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit(train_generator, validation_data=val_generator, epochs=20)

    model.save("xray_model.h5")
    print(f"✅ Modelo salvo")

    return model, train_generator.class_indices

def trainModelCats():
    _URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'

    path_to_zip = tf.keras.utils.get_file('cats_and_dogs_filtered.zip', origin=_URL, extract=False)

    # Extrair
    zip_dir = os.path.splitext(path_to_zip)[0]
    if not os.path.exists(zip_dir):
        with zipfile.ZipFile(path_to_zip, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(path_to_zip))

    # Caminhos
    base_dir = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')

    train_datagen = ImageDataGenerator(rescale=1. / 255)
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(150, 150), batch_size=20, class_mode='binary'
    )

    validation_generator = val_datagen.flow_from_directory(
        validation_dir, target_size=(150, 150), batch_size=20, class_mode='binary'
    )

    # Modelo CNN
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Treinamento
    model.fit(train_generator, epochs=3, validation_data=validation_generator)

    # Avaliação
    loss, acc = model.evaluate(validation_generator)
    print(f"Acurácia no conjunto de validação: {acc:.2%}")

    # Salvar o modelo treinado
    model.save("cats_vs_dogs_model.h5")
    print("✅ Modelo salvo em 'cats_vs_dogs_model.h5'")


def runAiCatsDogs(imagePath):
    model = tf.keras.models.load_model("cats_vs_dogs_model.h5")

    img_path = imagePath

    img = load_img(img_path, target_size=(150, 150))  # redimensiona
    img_array = img_to_array(img) / 255.0  # normaliza
    img_array = np.expand_dims(img_array, axis=0)  # adiciona dimensão batch

    prediction = model.predict(img_array)

    #   TODO: Gerar matriz com o resultado final do treinamento
    #         Separar as classes em anos ao invés de meses
    #

    if prediction[0][0] > 0.5:
        print("🐱 É um GATO com confiança:", prediction[0][0])
    else:
        print("🐶 É um CACHORRO com confiança:", 1 - prediction[0][0])

    return "gato" if prediction[0][0] > 0.5 else "cachorro"


def runAi(imagePath):
    model = tf.keras.models.load_model("xray_model.h5")

    img_path = imagePath

    img = load_img(img_path, target_size=(224, 224))  # redimensiona
    img_array = img_to_array(img) / 255.0  # normaliza
    img_array = np.expand_dims(img_array, axis=0)  # adiciona dimensão batch

    prediction = model.predict(img_array)

    #   TODO: Gerar matriz com o resultado final do treinamento
    #

    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)

    predicted_age_years = predicted_class

    print(f"📸 Classe predita: {predicted_class} (≈ {predicted_age_years} anos)")
    print(f"Confiança: {confidence:.2f}")

    return predicted_age_years, confidence
