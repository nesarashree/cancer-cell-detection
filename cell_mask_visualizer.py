'''
This Streamlit script visualizes cancel cell (CD33) mask matrices from CSV files by mapping grayscale intensity values to enlarged pixel blocks, 
enabling inspection of microscopy-derived segmentation data.
'''

import streamlit as st
import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# Load Matrix Data
# load a CSV file containing grayscale cell mask values
csv_path = "/Users/nesara/Desktop/CellDataViewer/CSV_from_cellmask_mat/W01_02_Mask_mat2csv.csv"
matrix_data = pd.read_csv(csv_path)

# Show matrix size and dimensions in the Streamlit app
st.write("Matrix data size:", matrix_data.size)
st.write(f"Matrix dimensions: {matrix_data.shape[0]} rows × {matrix_data.shape[1]} columns")

# OPTIONAL: preview the raw matrix values
st.write("Matrix preview:")
st.dataframe(matrix_data)

# Image Config
cell_pixels = 10  # Size of each cell block in pixels (was 2000, now more reasonable)
matrix_width, matrix_height = matrix_data.shape
image_width = matrix_height * cell_pixels
image_height = matrix_width * cell_pixels

# Initialize a black RGB image (OpenCV uses BGR format)
image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

# Draw the Matrix: each cell's intensity determines the grayscale color of the corresponding rectangle
# iterate over the matrix cells
for row in range(matrix_width):
    for col in range(matrix_height):
        value = matrix_data.iat[row, col]
        value = np.clip(int(value), 0, 255)  # Ensure valid grayscale range

        top_left = (col * cell_pixels, row * cell_pixels)
        bottom_right = ((col + 1) * cell_pixels, (row + 1) * cell_pixels)

        color = (value, value, value)  # Grayscale RGB triplet
        # visualize each matrix entry as a grayscale-filled rectangle using OpenCV
        cv2.rectangle(image, top_left, bottom_right, color, thickness=-1)

# Display the Image: use matplotlib to render the OpenCV-generated image in Streamlit
plt.figure(figsize=(10, 10))
plt.axis("off")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
st.pyplot(plt)
