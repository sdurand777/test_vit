
import torch
from PIL import Image
import torchvision.transforms as T
import timm  # PyTorch Image Models
import matplotlib.pyplot as plt
import numpy as np

# Charger le modèle ViT pré-entraîné depuis la bibliothèque timm
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=1000)
model.eval()

# Prétraitement de l'image pour le modèle ViT
transform = T.Compose([
    T.Resize((224, 224)),  # Redimensionner l'image à 224x224 pixels
    T.ToTensor(),  # Convertir l'image en tenseur
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normaliser
])

# Charger et prétraiter l'image
img = Image.open('images.jpeg')  # Remplacer par le chemin de votre image
img_tensor = transform(img).unsqueeze(0)  # Ajouter une dimension batch

# Découper l'image en patches et afficher les patches
def show_image_patches(image, patch_size=16):
    img_np = np.array(image)
    patches = []
    for i in range(0, img_np.shape[0] - patch_size + 1, patch_size):
        for j in range(0, img_np.shape[1] - patch_size + 1, patch_size):
            patch = img_np[i:i + patch_size, j:j + patch_size, :]
            patches.append(patch)

    fig, axes = plt.subplots(nrows=8, ncols=8, figsize=(8, 8))
    for ax, patch in zip(axes.flatten(), patches):
        ax.imshow(patch)
        ax.axis('off')
    plt.suptitle("Patches de l'image")
    plt.show()

# Afficher les patches de l'image originale
show_image_patches(img)

# Fonction pour extraire les cartes d'attention
def get_attention_maps(model, img_tensor):
    attn_maps = []

    def hook(module, input, output):
        if isinstance(module, timm.models.vision_transformer.VisionTransformer):
            attn_maps.append(output)

    # Ajouter le hook à la dernière couche d'attention
    hook_handle = model.blocks[-1].attn.attn_drop.register_forward_hook(hook)

    with torch.no_grad():
        _ = model(img_tensor)

    hook_handle.remove()
    return attn_maps

# Obtenir les cartes d'attention
attn_maps = get_attention_maps(model, img_tensor)

# Visualiser la carte d'attention de la dernière couche
def plot_attention_map(attn_map, num_heads=12):
    # Moyenne des têtes d'attention
    avg_attn_map = attn_map.mean(dim=1).squeeze().cpu().numpy()
    plt.imshow(avg_attn_map, cmap='viridis')
    plt.colorbar()
    plt.title("Carte d'attention de la dernière couche")
    plt.axis('off')
    plt.show()

# Visualiser la carte d'attention de la dernière couche
if attn_maps:
    plot_attention_map(attn_maps[-1])
else:
    print("Aucune carte d'attention disponible.")
