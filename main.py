
import torch
from PIL import Image
import torchvision.transforms as T
import timm  # PyTorch Image Models
import requests
import json

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

# Télécharger le fichier de labels d'ImageNet
url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
response = requests.get(url)
labels = response.json()

# Faire la prédiction
with torch.no_grad():
    output = model(img_tensor)

# Obtenir l'indice de la classe prédite
pred_idx = torch.argmax(output, dim=1).item()

# Récupérer le nom de la classe
pred_label = labels[pred_idx]
print(f"Classe prédite : {pred_label} (Indice : {pred_idx})")
