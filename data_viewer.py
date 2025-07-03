'''
This Streamlit app enables interactive visualization of multi-channel microscopy images alongside their cell segmentation ground truths. 
Users select a dataset folder containing cancer cell (CD33) images, then pick two specific channels to display: DIC, CD33, Ki67, PARP, CD3, Nucleus. 
It shows the brightened individual channels, a combined overlay highlighting overlapping signals in yellow, and the corresponding cell segmentation CSV as a grayscale mask. 
'''

import pandas as pd
import os
import streamlit as st
from PIL import Image
import numpy as np
from matplotlib.image import imread

# Constants for UI and color channels
IMAGE_PREVIEW_SIZE = 300
RED = 0
GREEN = 1
BLUE = 2
DS_STORE_FILENAME = '.DS_Store'

# Root folders for images and CSVs
DATA_IMAGES_ROOT = 'CellJpegs'
DATA_CSVS_ROOT = 'CellCSVs'

def get_clean_folder_list(folder_path):
    """Return sorted list of folder contents excluding .DS_Store"""
    items = os.listdir(folder_path)
    if DS_STORE_FILENAME in items:
        items.remove(DS_STORE_FILENAME)
    items.sort()
    return items

# Load folder lists, removing .DS_Store if present
image_folders = get_clean_folder_list(DATA_IMAGES_ROOT)
csv_files = get_clean_folder_list(DATA_CSVS_ROOT)

# Sidebar folder selection slider (1-based index)
selected_folder_idx = st.sidebar.slider(
    'Cell Image Folder', 1, len(image_folders), 1
) - 1
selected_folder_name = image_folders[selected_folder_idx]
selected_folder_path = os.path.join(DATA_IMAGES_ROOT, selected_folder_name)

# Skip folders starting with '.' (hidden/system folders)
if selected_folder_name.startswith('.'):
    st.warning(f"Skipping folder '{selected_folder_name}' (hidden or invalid)")
else:
    # List image files in the selected folder
    image_files = get_clean_folder_list(selected_folder_path)

    # Sliders for selecting channel images
    channel_a_idx = st.sidebar.slider(
        'Image A Cell Channel', 1, len(image_files), 1
    ) - 1
    channel_b_idx = st.sidebar.slider(
        'Image B Cell Channel', 1, len(image_files), min(2, len(image_files))
    ) - 1

    channel_a_filename = image_files[channel_a_idx]
    channel_b_filename = image_files[channel_b_idx]

    # Brightness multiplier slider
    brightness = st.sidebar.slider('Brightness Multiplier', 1, 10, 1)

    st.write(f"Selected Folder: **{selected_folder_name}**")
    st.write(f"Brightness Multiplier: **x{brightness}**")

    def load_and_brighten_image(img_path, channel):
        """Load image, multiply brightness, place in specified color channel"""
        img_data = imread(img_path).astype(np.float32) * brightness
        height, width = img_data.shape[:2]
        bright_img = np.zeros((height, width, 3), dtype=np.uint8)
        # Clip values and assign to color channel
        bright_img[:, :, channel] = np.clip(img_data, 0, 255).astype(np.uint8)
        return bright_img

    # Load and brighten images for channels A and B
    bright_img_a = load_and_brighten_image(os.path.join(selected_folder_path, channel_a_filename), RED)
    bright_img_b = load_and_brighten_image(os.path.join(selected_folder_path, channel_b_filename), GREEN)

    # Mapping for channel labels (assuming filenames are consistent, see cell_JPG_images folder for naming reference)
    channel_labels = {
        0: "Channel 1: DIC",
        1: "Channel 2: CD33",
        2: "Channel 3: Ki67",
        3: "Channel 4: PARP",
        4: "Channel 5: CD3",
        5: "Channel 6: Nucleus"
    }

    # Show images and their labels
    st.subheader(f"Channel A: {channel_labels.get(channel_a_idx, 'Unknown')}")
    st.image(bright_img_a, width=IMAGE_PREVIEW_SIZE)

    st.subheader(f"Channel B: {channel_labels.get(channel_b_idx, 'Unknown')}")
    st.image(bright_img_b, width=IMAGE_PREVIEW_SIZE)

    # Combine channels visually (yellow = overlap)
    combined = np.clip(bright_img_a.astype(np.uint16) + bright_img_b.astype(np.uint16), 0, 255).astype(np.uint8)
    st.subheader("Combined Channels (Yellow = Overlap)")
    st.image(combined, width=IMAGE_PREVIEW_SIZE)

    # Show Ground Truth from corresponding CSV file
    st.subheader("Cell CSV Ground Truth")

    # Map image folder index to CSV file
    csv_filename = csv_files[selected_folder_idx]
    csv_filepath = os.path.join(DATA_CSVS_ROOT, csv_filename)
    cell_csv = pd.read_csv(csv_filepath)

    # Convert CSV to image format and apply brightness scaling
    cell_csv_scaled = np.clip(cell_csv * brightness, 0, 1) * 200
    cell_csv_img = cell_csv_scaled.astype(np.uint8)

    st.image(cell_csv_img, width=800)
