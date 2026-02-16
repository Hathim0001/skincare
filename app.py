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
