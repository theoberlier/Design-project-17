#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 14:21:35 2025

@author: mariusneamtu
"""


import os
import cv2

name = "10cm_2m_10" # +"_bis"

# Input and output directories
input_dir = name+"_bis"+"/"+name+"_cropped2"
output_dir = name+"_bis"+"/"+name+"_resized2"

# Scaling factor (e.g., 0.5 = reduce size by 50%)
scale = 0.5

# Create output folder if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Walk through the input directory and process .jpg files
for root, _, files in os.walk(input_dir):
    for file in files:
        if file.lower().endswith('.jpg'):
            input_path = os.path.join(root, file)
            output_path = os.path.join(output_dir, file)

            # Read the image
            image = cv2.imread(input_path)
            if image is None:
                print(f"Error reading image: {input_path}")
                continue

            # Calculate new size based on scale
            height, width = image.shape[:2]
            new_size = (int(width * scale), int(height * scale))

            # Resize the image
            resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

            # Save the resized image to the output folder
            cv2.imwrite(output_path, resized_image)
            print(f"Saved image: {output_path}")