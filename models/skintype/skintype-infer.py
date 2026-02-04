import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

model = torch.load('saved_models/skintype/best_20.pt')
model.eval()

def predict_skin_type(x):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(x).convert("RGB")
    img = preprocess(np.array(img))
    img = img.view(1, 3, 224, 224)
    with torch.no_grad():
        out = model(img)
        return out.argmax(1).item()
