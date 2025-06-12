#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 14:40:52 2025

@author: mariusneamtu
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Folder paths
input_folder = '7cm_2m_5/7cm_2m_5_bw'
output_folder = '7cm_2m_5/7cm_2m_5_final'

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

all_curve = []

for filename in os.listdir(input_folder):
    if filename.lower().endswith('.tif'):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename.replace('.tif', '_processed.png'))
        
        # Load the image in grayscale
        img_gray = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        
        height, width = img_gray.shape
        step = 5  # Analyze one column every 5 pixels
        points_air_water = []
        points_sediment_water = []
        
        # runs accros colums
        for x in range(0, width, step):
            column = img_gray[:, x].astype(np.float32)
            column = cv2.GaussianBlur(column[:, np.newaxis], (11, 1), 0).squeeze()
           
            # Compute the difference between consecutive pixels
            diff = np.diff(column)
            
            # Identify the highest positive jump (dark -> light) for the free surface of water
            #air_to_water_idx = np.argmax(diff[1:-1]) + 2

            # Identify the highest negative jump (light -> dark) for the sediment-water interface
            #water_to_sediment_idx = np.argmin(diff[1:-1]) + 2

            air_to_water_idx = len(diff[1:-1]) - np.argmax(diff[1:-1][::-1]) + 1
            water_to_sediment_idx = len(diff[1:-1]) - np.argmin(diff[1:-1][::-1]) + 1
            
            # Ensure the interfaces are in the correct order
            if 0 < air_to_water_idx < height - 1 and 0 < water_to_sediment_idx < height - 1:
                if air_to_water_idx < water_to_sediment_idx:
                    points_air_water.append((x, air_to_water_idx))
                    points_sediment_water.append((x, water_to_sediment_idx))
        
        # Plot the image with the detected points
        plt.figure(figsize=(10, 6))
        plt.imshow(img_gray, cmap='gray')
        
        if points_air_water:
            points_air_water = np.array(points_air_water)
            plt.plot(points_air_water[:, 0], points_air_water[:, 1], 'b-', linewidth=2, label='Water Free Surface')
        
        if points_sediment_water:
            points_sediment_water = np.array(points_sediment_water)
            plt.plot(points_sediment_water[:, 0], points_sediment_water[:, 1], 'r-', linewidth=2, label='Sediment-Water Interface')
        
        if len(points_air_water) > 0 or len(points_sediment_water) > 0:
            plt.legend()
        
        plt.axis('off')
        #plt.savefig(output_path, bbox_inches='tight')
        plt.close()

        all_curve.append(points_sediment_water)

        #print(f'Processed {filename}')


###############################################################
#plan 3D avec les bonnes valeurs brutes
###############################################################

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors
from scipy.interpolate import griddata

fig = plt.figure(figsize=(12, 8))  # Augmenter la taille de la figure
ax = fig.add_subplot(111, projection='3d')

# Collecte des points X, Y et Temps
x_vals, y_vals, t_vals = [], [], []

for time_index, points in enumerate(all_curve):
    points = points[20:]
    #if len(points) > 0 and time_index<20:   # for 4cm and 1.5m wabes
    if len(points) > 0 :#and time_index<70:
        points = np.array(points)
        x_vals.extend(points[:, 0])
        y_vals.extend(points[:, 1])
        t_vals.extend([time_index] * len(points))

x_vals = np.array(x_vals)*115/1875
y_vals = np.array(y_vals)*115/1875
t_vals = np.array(t_vals)/2

# Création d'une grille régulière pour interpolation
grid_x, grid_t = np.meshgrid(
    np.linspace(np.min(x_vals), np.max(x_vals), 100),  # 100 points en X
    np.linspace(np.min(t_vals), np.max(t_vals), 50)   # 50 points en temps
)

# Interpolation de Y sur la grille
grid_y = griddata((x_vals, t_vals), y_vals, (grid_x, grid_t), method='cubic')

# Tracé de la surface lissée
surf = ax.plot_surface(grid_x, grid_t, grid_y, cmap='PuBu', edgecolor='none')

ax.set_xlabel('Position horizontale [cm]', labelpad=40, fontsize=18)
ax.set_ylabel('Temps [min]', fontsize=18)
ax.set_zlabel('Interface Eau-Sédiment [cm]', labelpad=20, fontsize=18)
#ax.set_title('Évolution Lissée de l’Interface Eau-Sédiment')

# **Inverser les axes comme demandé**
ax.invert_xaxis()  # X = 0 à droite
ax.invert_zaxis()  # Y = 0 en haut

# **Conserver des proportions réalistes**
scale_x = np.ptp(x_vals)  # Étendue réelle de X
scale_y = np.ptp(y_vals)  # Étendue réelle de Y
scale_t = np.ptp(t_vals)  # Étendue du temps

ax.set_box_aspect([scale_x, scale_y, scale_y])  # Appliquer le bon ratio

# Ajouter une barre de couleur
#cbar = plt.colorbar(surf, ax=ax, shrink=0.6, aspect=10)
#cbar.set_label('Hauteur de l’Interface Eau-Sédiment')

plt.show()


########################################################
#Courbe en 4 temps 
########################################################

import matplotlib.pyplot as plt
import numpy as np

# On extrait les time_index disponibles
time_indices = [i for i, points in enumerate(all_curve) if len(points) > 0]

# Vérification que l'on a bien au moins 4 étapes dans le temps
if len(time_indices) >= 4:
    # Déterminons les étapes clés
    first_index = time_indices[0]
    last_index = time_indices[-1]
    quarter_index = time_indices[len(time_indices) // 4]
    three_quarter_index = time_indices[(3 * len(time_indices)) // 4]

    selected_indices = [first_index, quarter_index, three_quarter_index, last_index]
    labels = ['Début', '1/4 du temps', '3/4 du temps', 'Fin']
    colors = ['blue', 'green', 'orange', 'red']

    plt.figure(figsize=(10, 6))

    for idx, label, color in zip(selected_indices, labels, colors):
        points = np.array(all_curve[idx])
        if len(points) > 0:
            x = points[:, 0]
            y = points[:, 1]
            plt.plot(x, y, label=label, color=color)

    plt.xlabel('Position X')
    plt.ylabel('Hauteur Y')
    plt.title('Évolution de l\'interface Eau-Sédiment')
    plt.legend()
    plt.grid(True)
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()  # si tu veux la même orientation que le 3D
    plt.show()

else:
    print("Pas assez de données pour tracer 4 courbes.")

