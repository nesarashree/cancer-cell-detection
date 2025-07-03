'''
This script processes cell mask CSVs and cancer-positive cell lists to compute bounding boxes for CD33+ cancer cells, 
generating XML annotations in Pascal VOC format for object detection training.
''''
import os
import pandas as pd

# Constants and parameters
MAX_NUM_CELLS = 1500
Y_CORRECTION = -12
X_CORRECTION = 7
CROP_X_TL = 0
CROP_Y_TL = 0
CROP_X_BR = 800
CROP_Y_BR = 800
DS_STORE = '.DS_Store'

# Input directories
CELL_CSV_FOLDER = 'CellCSVs'
CANCER_CSV_FOLDER = 'CD33+ cells'

# Helper function to remove unwanted files like .DS_Store and sort lists
def clean_and_sort_folder_list(folder_path):
    file_list = os.listdir(folder_path)
    if DS_STORE in file_list:
        file_list.remove(DS_STORE)
    file_list.sort()
    return file_list

# Load and clean file lists
cell_csv_files = clean_and_sort_folder_list(CELL_CSV_FOLDER)
cancer_csv_files = clean_and_sort_folder_list(CANCER_CSV_FOLDER)

# Process each well/sample
for well_index in range(len(cell_csv_files)):
    print("\n" + "_"*100)
    print(f"Processing well {well_index + 1}")

    # Load cancer-positive cell IDs into a dictionary for quick lookup
    cancer_csv_path = os.path.join(CANCER_CSV_FOLDER, cancer_csv_files[well_index])
    cancer_positive_dict = {}
    with open(cancer_csv_path, 'r') as f:
        next(f)  # skip header line
        for line in f:
            line = line.strip()
            cancer_cell_num, _ = line.split(',')
            cancer_positive_dict[int(cancer_cell_num)] = True
    print(f"Loaded cancer positive dictionary from {cancer_csv_files[well_index]}")

    # Load cell mask CSV matrix
    cell_csv_path = os.path.join(CELL_CSV_FOLDER, cell_csv_files[well_index])
    cell_matrix = pd.read_csv(cell_csv_path).to_numpy()
    matrix_height, matrix_width = cell_matrix.shape
    print(f"{cell_csv_files[well_index]} size: {matrix_height} rows x {matrix_width} cols\n")

    # Initialize bounding box data structures
    smallest_x = [-1] * MAX_NUM_CELLS
    smallest_y = [-1] * MAX_NUM_CELLS
    largest_x = [-1] * MAX_NUM_CELLS
    largest_y = [-1] * MAX_NUM_CELLS
    found_cell = [False] * MAX_NUM_CELLS
    num_found_cells = 0

    # Calculate bounding boxes for each cell by scanning the matrix
    for y in range(matrix_height):
        if y % 100 == 0:
            progress = (y / matrix_height) * 100
            print(f"Progress: {progress:.2f}%")
        for x in range(matrix_width):
            cell_id = cell_matrix[y, x]
            if cell_id == 0:
                continue

            # Update bounding box coordinates
            if smallest_x[cell_id] == -1 or x < smallest_x[cell_id]:
                smallest_x[cell_id] = x
            if smallest_y[cell_id] == -1 or y < smallest_y[cell_id]:
                smallest_y[cell_id] = y
            if x > largest_x[cell_id]:
                largest_x[cell_id] = x
            if y > largest_y[cell_id]:
                largest_y[cell_id] = y

            # Mark cell as found once all coords initialized
            if (not found_cell[cell_id] and 
                smallest_x[cell_id] >= 0 and smallest_y[cell_id] >= 0 and
                largest_x[cell_id] >= 0 and largest_y[cell_id] >= 0):
                found_cell[cell_id] = True
                num_found_cells += 1

    # Prepare XML output file name
    file_prefix = f"Cancer_W0{well_index + 1}" if well_index < 9 else f"Cancer_W{well_index + 1}"
    xml_filename = f"{file_prefix}_1.xml"
    print(f"Writing annotations to {xml_filename}")

    with open(xml_filename, "w") as xml_file:
        # Write XML header
        header = [
            "<annotation>\n",
            "\t<folder>training_images</folder>\n",
            f"\t<filename>{file_prefix}_6.jpg</filename>\n",
            f"\t<path>/root/data/{file_prefix}_6.jpg</path>\n",
            "\t<source>\n",
            "\t\t<database>Unknown</database>\n",
            "\t</source>\n",
            "\t<size>\n",
            "\t\t<width>800</width>\n",
            "\t\t<height>800</height>\n",
            "\t\t<depth>3</depth>\n",
            "\t</size>\n",
            "\t<segmented>0</segmented>\n"
        ]
        xml_file.writelines(header)

        cell_counter = 0

        # Write bounding boxes for cancer-positive cells within crop bounds
        for cell_id in range(1, num_found_cells):
            if not found_cell[cell_id]:
                continue
            if (smallest_x[cell_id] < CROP_X_TL or smallest_y[cell_id] < CROP_Y_TL or
                largest_x[cell_id] > CROP_X_BR or largest_y[cell_id] > CROP_Y_BR):
                continue
            if cell_id not in cancer_positive_dict:
                continue

            # Apply corrections and crop offsets
            min_x = smallest_x[cell_id] + X_CORRECTION - CROP_X_TL
            min_y = smallest_y[cell_id] + Y_CORRECTION - CROP_Y_TL
            max_x = largest_x[cell_id] + X_CORRECTION - CROP_X_TL
            max_y = largest_y[cell_id] + Y_CORRECTION - CROP_Y_TL

            # Clamp values to image boundaries
            min_x = max(0, min(min_x, CROP_X_BR - CROP_X_TL))
            max_x = max(0, min(max_x, CROP_X_BR - CROP_X_TL))
            min_y = max(0, min(min_y, CROP_Y_BR - CROP_Y_TL))
            max_y = max(0, min(max_y, CROP_Y_BR - CROP_Y_TL))

            # Write XML object block
            obj_xml = [
                "\t<object>\n",
                "\t\t<name>cancerCell</name>\n",
                "\t\t<pose>Unspecified</pose>\n",
                "\t\t<truncated>0</truncated>\n",
                "\t\t<difficult>0</difficult>\n",
                "\t\t<bndbox>\n",
                f"\t\t\t<xmin>{min_x}</xmin>\n",
                f"\t\t\t<ymin>{min_y}</ymin>\n",
                f"\t\t\t<xmax>{max_x}</xmax>\n",
                f"\t\t\t<ymax>{max_y}</ymax>\n",
                "\t\t</bndbox>\n",
                "\t</object>\n"
            ]
            xml_file.writelines(obj_xml)
            cell_counter += 1

        # Write XML footer
        xml_file.write("</annotation>\n")

    print(f"Annotated {cell_counter} cancer cells in {xml_filename}")

