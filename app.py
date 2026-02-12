import streamlit as st
import numpy as np
import pandas as pd
import torch
from PIL import Image

st.title("Skin Care Essentials")
st.header("Personalized Skin Care Recommendations")

products_path = 'recommender/essential_skin_care.csv'
features = [
    'normal', 'dry', 'oily', 'combination', 'sensitive', 'general care',
    'hydration', 'dull skin', 'dryness', 'softening', 'smoothening',
    'fine lines', 'wrinkles', 'acne', 'blemishes', 'pore care',
    'daily use', 'dark circles'
]
