#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 12:32:48 2025

@author: mariusneamtu
"""


import os
from PIL import Image

name = "10cm_2m_10" #   "_bis"+

# Folder paths
input_folder = name+"_bis"+"/"+name+"_input2"
output_folder = name+"_bis"+"/"+name+"_cropped2"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Crop coordinates
x, y, width, height = 126, 1700, 5000, 1700

# Iterate over all files in the folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
        # Full file paths
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        try:
            # Open the image
            with Image.open(input_path) as img:
                # Perform the crop
                cropped_img = img.crop((x, y, x + width, y + height))
                # Save the cropped image
                cropped_img.save(output_path)
                print(f"Saved: {output_path}")
        except Exception as e:
            print(f"Error with {filename}: {e}")

print("Processing completed!")