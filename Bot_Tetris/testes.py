import cv2
import os
import numpy as np

# Definindo a classe para os tetrominos (apenas como exemplo)
CLASSES = ['I', 'O', 'T', 'S', 'Z', 'L', 'J']

def load_images_from_directory(directory):
    """Carrega imagens do diretório e verifica se são carregadas corretamente."""
    image_paths = []
    labels = []

    for label, class_name in enumerate(CLASSES):
        class_folder = os.path.join(directory, class_name)
        if not os.path.exists(class_folder):
            print(f'Pasta de classe não encontrada: {class_folder}')
            continue

        for filename in os.listdir(class_folder):
            if filename.endswith(".png"):
                img_path = os.path.join(class_folder, filename)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Erro ao carregar imagem: {img_path}")
                else:
                    image_paths.append(img_path)
                    labels.append(label)

    print(f'Imagens carregadas: {len(image_paths)}')
    return image_paths, labels

def resize_image_if_needed(img, target_size):
    """Redimensiona a imagem para o tamanho desejado, caso seja necessário."""
    height, width = img.shape[:2]
    if height < target_size[0] or width < target_size[1]:
        img = cv2.resize(img, (target_size[1], target_size[0]))  # Redimensiona para o tamanho alvo
    return img

def pad_images(images, target_size):
    """Aplica padding às imagens para que todas tenham o mesmo tamanho."""
    padded_images = []
    for img in images:
        if img is None:
            print("Imagem não carregada corretamente, saltando...")
            continue

        # Verificar as dimensões da imagem
        print(f"Forma da imagem original: {img.shape}")

        # Garantir que a imagem tem 3 canais (RGB)
        if len(img.shape) == 2:  # Imagem em escala de cinza (1 canal)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Garantir que a imagem tem as dimensões corretas
        if len(img.shape) != 3 or img.shape[2] != 3:
            print(f"Imagem com formato inesperado (esperado 3 canais RGB): {img.shape}")
            continue

        # Redimensionar se necessário
        img = resize_image_if_needed(img, target_size)

        # Ajustar a forma da imagem
        height, width = img.shape[:2]
        top = bottom = (target_size[0] - height) // 2
        left = right = (target_size[1] - width) // 2

        # Adicionar bordas (padding)
        padded_image = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        padded_images.append(padded_image)

    return padded_images

def main():
    # Caminho do diretório com as imagens (ajustar conforme necessário)
    directory = r"dataset\manual"
    target_size = (128, 128)  # Definindo o tamanho de saída das imagens (exemplo: 128x128)

    # Carregar as imagens e verificar
    image_paths, labels = load_images_from_directory(directory)

    # Carregar e aplicar padding nas imagens
    images = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        images.append(img)

    padded_images = pad_images(images, target_size)

    # Verificar a forma das imagens após padding
    if len(padded_images) > 0:
        print(f"Forma das imagens após padding: {padded_images[0].shape}")
    else:
        print("Nenhuma imagem foi carregada corretamente.")

if __name__ == "__main__":
    main()
