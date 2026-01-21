import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image

# 1. Cargar VGG19 (solo las capas de características)
vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.eval()

# 2. Función para procesar la imagen
def load_image(img_path, max_size=400):
    image = Image.open(img_path).convert('RGB')
    size = max_size if max(image.size) > max_size else max(image.size)
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return transform(image).unsqueeze(0)

# 3. Cálculo de la matriz de Gram (Fundamental para el estilo)
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    return torch.mm(tensor, tensor.t())

print("Script de transferencia de estilo listo.")