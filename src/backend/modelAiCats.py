from . import modelAi

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
    print("Modelo salvo em 'cats_vs_dogs_model.h5'")

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
        print("É um GATO com confiança:", prediction[0][0])
    else:
        print("É um CACHORRO com confiança:", 1 - prediction[0][0])

    return "gato" if prediction[0][0] > 0.5 else "cachorro"