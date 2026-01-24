import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from acne_model import MyNet

model = torch.load('saved_models/acne-severity/best_6.pt')
model.eval()

def predict_acne_level(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    class_name = ['Minimal', 'Moderate', 'Severe', 'Extreme']
    image = Image.open(image_path)
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = model(input_batch)
    prob = nn.Softmax(dim=1)
    probs = prob(output)
    predicted_class = torch.argmax(probs, axis=1).tolist()[0]
    return class_name[predicted_class]
