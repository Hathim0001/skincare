# Web scraper for beauty and skincare catalogs
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

categories = ['face-moisturisers', 'cleanser', 'sunscreen', 'concealer', 'mask-and-peel', 'foundation']
products_from_each_category = 300
pages = products_from_each_category // 50
df = pd.DataFrame(columns=['label', 'url'])
