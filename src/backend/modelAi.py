import os
import zipfile
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


print(tf.config.list_physical_devices('GPU'))

def trainModel():
    #for pure windows
    #dataPath = r"C:\Users\guizo\Downloads\Bone+Age+Training+Set+Annotations\train.csv"
    #image_dir = r"C:\Users\guizo\Downloads\Bone+Age+Training+Set\boneage-training-dataset"

    #for wsl build
    dataPath = "/mnt/c/Users/guizo/Downloads/Bone+Age+Training+Set+Annotations/train.csv"
    image_dir = "/mnt/c/Users/guizo/Downloads/Bone+Age+Training+Set/boneage-training-dataset"

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
        class_mode="categorical",
        #classes=all_classes,
        shuffle=True
    )

    val_generator = datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=image_dir,
        x_col="id",
        y_col="boneage_class",
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
        #classes=all_classes,
        shuffle=False
    )

    print("INFO: Imagens no treino:", train_generator.samples)
    print("INFO: Imagens na validação:", val_generator.samples)

    num_classes = len(train_generator.class_indices)
    print("INFO: Número de classes detectadas:", num_classes)


    classes_in_train = np.unique(train_df["boneage_class"].values)

    # Calcula os pesos
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes_in_train,
        y=train_df["boneage_class"].values
    )

    class_weight_dict = dict(zip(classes_in_train, class_weights))
    print("Class weights:", class_weight_dict)

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

    model.fit(train_generator, validation_data=val_generator, epochs=8, class_weight=class_weight_dict)

    model.save("xray_model.h5")
    print(f"Modelo salvo")



    y_pred_probs = model.predict(val_generator)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = val_generator.classes

    class_labels = sorted([int(k) for k in val_generator.class_indices.keys()])
    class_names = [str(k) for k in class_labels]

    # --- Classification report ---
    report = classification_report(
        y_true, 
        y_pred, 
        labels=class_labels,
        target_names=class_names
    )
    print("Classification Report:\n", report)

    with open("classification_report.txt", "w") as f:
        f.write(report)
    print("Classification report salvo em classification_report.txt")

    # --- Matriz de confusão ---
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)

    plt.figure(figsize=(12,8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Matriz de Confusão")
    plt.savefig("confusion_matrix.png")
    plt.close()
    print("Matriz de confusão salva em confusion_matrix.png")

    return model, train_generator.class_indices


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

    print(f"Classe predita: {predicted_class} (≈ {predicted_age_years} anos)")
    print(f"Confiança: {confidence:.2f}")

    return predicted_age_years, confidence