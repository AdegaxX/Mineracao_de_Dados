import cv2
import numpy as np
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Função para preencher as imagens para um tamanho fixo
def pad_images(images, target_size):
    padded_images = []
    for img in images:
        if img is None or len(img.shape) != 3 or img.shape[2] != 3:
            print(f"Imagem inválida encontrada: {img}")
            continue

        if img.shape[0] != target_size[0] or img.shape[1] != target_size[1]:
            top = (target_size[0] - img.shape[0]) // 2
            bottom = target_size[0] - img.shape[0] - top
            left = (target_size[1] - img.shape[1]) // 2
            right = target_size[1] - img.shape[1] - left

            if top < 0 or bottom < 0 or left < 0 or right < 0:
                print(f"Padding inválido para a imagem com tamanho {img.shape}")
                continue

            padded_image = cv2.copyMakeBorder(
                img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
        else:
            padded_image = img  # Imagem já no tamanho correto

        padded_images.append(padded_image)

    return np.array(padded_images)


# Função para carregar e processar as imagens
def load_and_process_images(directory, target_size):
    image_paths = []
    labels = []

    print(f"Verificando o diretório: {directory}")

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg"):
                image_paths.append(os.path.join(root, file))
                label = root.split(os.path.sep)[-1]
                labels.append(label)

    print(f"Imagens encontradas: {len(image_paths)}")

    X = []
    y = []
    for img_path, label in zip(image_paths, labels):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Erro ao carregar imagem: {img_path}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img_to_array(img)
        X.append(img)
        y.append(label)

        print(f"Forma da imagem carregada: {img.shape}")

    if len(X) == 0:
        print("Erro: Nenhuma imagem foi carregada.")
        return None, None

    # Padroniza as imagens para o mesmo tamanho
    print("Aplicando padding...")
    X = pad_images(X, target_size)

    if X is not None and X.size > 0:
        print(f"Forma das imagens após padding: {X.shape}")
    else:
        print("Erro ao aplicar padding nas imagens.")
        return None, None

    # Verificando se o número de amostras de X e y coincide
    if len(X) != len(y):
        print(f"Erro: O número de imagens ({len(X)}) não corresponde ao número de rótulos ({len(y)}).")
        return None, None

    return X, y


# Função para pré-processar os rótulos (one-hot encoding)
def preprocess_labels(y):
    if len(y) == 0:
        print("Erro: Nenhum rótulo para processar.")
        return None

    label_map = {label: idx for idx, label in enumerate(set(y))}
    y_encoded = np.array([label_map[label] for label in y])
    return to_categorical(y_encoded)


# Parâmetros
directory = "dataset/testes"  # Caminho para o diretório onde estão as imagens
target_size = (128, 128)  # Tamanho de destino das imagens (altura, largura)

# Carregando e processando as imagens
X, y = load_and_process_images(directory, target_size)

if X is None or y is None:
    print("Erro ao processar imagens. Verifique o diretório ou o formato das imagens.")
else:
    # Pré-processando os rótulos (se necessário)
    y = preprocess_labels(y)

    # Criando o modelo com 7 classes de saída
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(target_size[0], target_size[1], 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(7, activation='softmax')  # Agora com 7 neurônios (para 7 classes)
    ])

    # Compilando o modelo
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Treinando o modelo
    model.fit(X, y, epochs=10, batch_size=32)

    # Salvando o modelo treinado
    model.save('modelo_tetromino.keras')

    # Previsões no conjunto de testes
    y_pred = model.predict(X)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Pega as classes previstas

    # Matriz de confusão
    cm = confusion_matrix(np.argmax(y, axis=1), y_pred_classes)

    # Visualizando a matriz de confusão
    class_names = list(set(y.flatten()))  # As classes podem ser derivadas dos rótulos
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
