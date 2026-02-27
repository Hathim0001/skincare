import streamlit as st
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
from models.acne.acne_model import MyNet
from recommender.recommend import Recommender

products_path = 'recommender/essential_skin_care.csv'
features = [
    'normal', 'dry', 'oily', 'combination', 'sensitive', 'general care',
    'hydration', 'dull skin', 'dryness', 'softening', 'smoothening',
    'fine lines', 'wrinkles', 'acne', 'blemishes', 'pore care',
    'daily use', 'dark circles'
]

try:
    acne_model = MyNet()
    acne_model.load_state_dict(torch.load('saved_models/acne-severity/best_6.pt', map_location='cpu'))
    acne_model.eval()
except Exception:
    acne_model = None

try:
    skintype_model = MyNet()
    skintype_model.load_state_dict(torch.load('saved_models/skintype/best_20.pt', map_location='cpu'))
    skintype_model.eval()
except Exception:
    skintype_model = None

recommender = Recommender(products_path, features)

def predict_acne_level(image):
    if acne_model is None:
        return None
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = acne_model(input_tensor)
        probs = nn.Softmax(dim=1)(output)
        predicted_class = torch.argmax(probs, axis=1).item()
    return predicted_class

def predict_skin_type(img):
    if skintype_model is None:
        return None
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        output = skintype_model(input_tensor)
        predicted_class = torch.argmax(output, axis=1).item()
    return predicted_class

def get_feature_vector(skin_type, acne_level, selected_features):
    fv = [0] * len(features)
    if skin_type == 'Normal':
        fv[0] = 1
    elif skin_type == 'Dry':
        fv[1] = 1
    else:
        fv[2] = 1
    if acne_level >= 2:
        fv[13] = 1
    for feature in selected_features:
        fv[features.index(feature)] = 1
    return fv

def get_essential_skincare(fv):
    return recommender.recommend(fv)

def main():
    st.title("Skin Care Essentials")
    st.header("Personalized Skin Care Recommendations")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if acne_model is not None:
            acne_level = predict_acne_level(image)
            severity_levels = ['Clear', 'Mild', 'Severe', 'Extreme']
            severity = severity_levels[acne_level] if acne_level is not None else "Unable to Predict"
            st.subheader("Acne Severity Level")
            st.write(severity)
        else:
            st.subheader("Acne Severity Level")
            st.write("Model not loaded")

        if skintype_model is not None:
            skin_type_idx = predict_skin_type(image)
            skin_types = ['Normal', 'Dry', 'Oily']
            skin_type = skin_types[skin_type_idx] if skin_type_idx is not None else "Unable to Predict"
            st.subheader("Skin Type")
            st.write(skin_type)
        else:
            st.subheader("Skin Type")
            st.write("Model not loaded")
