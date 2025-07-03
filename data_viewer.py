'''
This Streamlit app visualizes cancer cell images from two selectable fluorescence channels with adjustable brightness, 
combines them to highlight overlapping regions, and displays the corresponding cell segmentation ground truth from CSV data.
'''

import os
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
from matplotlib.image import imread

# Constants for image preview size and color channel indices
IMAGE_PREVIEW_SIZE = 300
RED_CHANNEL = 0
GREEN_CHANNEL = 1
BLUE_CHANNEL = 2
DS_STORE_FILENAME = '.DS_Store'

# Root folders containing images and CSV data
DATA_IMAGES_ROOT = 'CellJpegs'
DATA_CSVS_ROOT = 'CellCSVs'

def get_sorted_folder_list(root_folder):
    """Return sorted list of folder contents excluding system files like .DS_Store."""
    items = os.listdir(root_folder)
    if DS_STORE_FILENAME in items:
        items.remove(DS_STORE_FILENAME)
    items.sort()
    return items

# Get list of image folders and CSV files
image_folders = get_sorted_folder_list(DATA_IMAGES_ROOT)
csv_files = get_sorted_folder_list(DATA_CSVS_ROOT)

# Sidebar selection for image folder
folder_index = st.sidebar.slider('Select Cell Image Folder', 1, len(image_folders)) - 1
selected_folder = image_folders[folder_index]
selected_folder_path = os.path.join(DATA_IMAGES_ROOT, selected_folder)

# Skip folders starting with '.' (hidden/system files)
if selected_folder.startswith('.'):
    st.write(f"Skipping folder: {selected_folder}")
else:
    # List image files inside the selected folder
    image_files = get_sorted_folder_list(selected_folder_path)

    # Sidebar sliders to select two image channels from the folder
    channel_a_index = st.sidebar.slider('Select Image Channel A', 1, len(image_files)) - 1
    channel_b_index = st.sidebar.slider('Select Image Channel B', 1, len(image_files)) - 1

    channel_a_name = image_files[channel_a_index]
    channel_b_name = image_files[channel_b_index]

    # Sidebar slider for brightness multiplier
    brightness_multiplier = st.sidebar.slider('Brightness Multiplier', 1, 10, 1)

    st.write(f"Image folder: {selected_folder}")
    st.write(f"Channel A: {channel_a_name}")
    st.write(f"Channel B: {channel_b_name}")
    st.write(f"Brightness multiplier: {brightness_multiplier}x")

    # Load and brighten Channel A image
    channel_a_path = os.path.join(selected_folder_path, channel_a_name)
    image_a = imread(channel_a_path).astype(np.float32) * brightness_multiplier

    # Prepare blank RGB image and assign brightened data to Red channel
    image_a_rgb = np.zeros((*image_a.shape[:2], 3), dtype=np.uint8)
    image_a_rgb[:, :, RED_CHANNEL] = np.clip(image_a, 0, 255).astype(np.uint8)

    # Load and brighten Channel B image
    channel_b_path = os.path.join(selected_folder_path, channel_b_name)
    image_b = imread(channel_b_path).astype(np.float32) * brightness_multiplier

    # Prepare blank RGB image and assign brightened data to Green channel
    image_b_rgb = np.zeros((*image_b.shape[:2], 3), dtype=np.uint8)
    image_b_rgb[:, :, GREEN_CHANNEL] = np.clip(image_b, 0, 255).astype(np.uint8)

    # Display Channel A
    st.subheader(f"Channel A: {channel_a_name}")
    st.image(image_a_rgb, width=IMAGE_PREVIEW_SIZE)

    # Display Channel B
    st.subheader(f"Channel B: {channel_b_name}")
    st.image(image_b_rgb, width=IMAGE_PREVIEW_SIZE)

    # Combine channels by adding their RGB arrays
    combined_rgb = np.clip(image_a_rgb + image_b_rgb, 0, 255).astype(np.uint8)

    st.subheader("Combined Channels (Yellow = Overlap)")
    st.image(combined_rgb, width=IMAGE_PREVIEW_SIZE)

    # Load and display the corresponding CSV ground truth as grayscale image
    csv_file_name = csv_files[folder_index]
    csv_path = os.path.join(DATA_CSVS_ROOT, csv_file_name)
    cell_csv = pd.read_csv(csv_path)

    cell_csv_image = (cell_csv * brightness_multiplier).clip(0, 1) * 200  # scale for visibility
    cell_csv_image = cell_csv_image.astype(np.uint8)

    st.subheader("Cell CSV Ground Truth")
    st.image(cell_csv_image, width=800)
