import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

# 1. Configuración de transformaciones (Pre-procesamiento)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 2. Carga de datos (Asumiendo que tienes carpetas 'train' y 'val')
# train_dataset = datasets.ImageFolder('path/to/data/train', transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 3. Cargar ResNet18 pre-entrenada
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Congelar todas las capas excepto la última
for param in model.parameters():
    param.requires_grad = False

# Reemplazar la última capa para 2 clases (ej: Perro vs Gato)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

# 4. Entrenamiento (Loop básico)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

print("Estructura del modelo cargada correctamente.")