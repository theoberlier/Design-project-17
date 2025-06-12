# 4*4 plot avec les différences de toutes les expériences 

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

sequence_folders = [
    '4cm_1.5m_5/4cm_1.5m_5_bw',
    '7cm_1.5m_5/7cm_1.5m_5_bw',
    '10cm_1.5m_5/10cm_1.5m_5_bw',
    '10cm_1.5m_5/10cm_1.5m_5_bw',#Il me manque cette image
    '4cm_2m_5/4cm_2m_5_bw',
    '7cm_2m_5/7cm_2m_5_bw',
    '10cm_2m_5/10cm_2m_5_bw',
    '10cm_2m_5_bis/10cm_2m_5_bw2',
    '4cm_1.5m_10/4cm_1.5m_10_bw',
    '7cm_1.5m_10/7cm_1.5m_10_bw',
    '10cm_1.5m_10/10cm_1.5m_10_bw',
    '10cm_1.5m_10_bis/10cm_1.5m_10_bw2',
    '4cm_2m_10/4cm_2m_10_bw',
    '7cm_2m_10/7cm_2m_10_bw',
    '10cm_2m_10/10cm_2m_10_bw',
    '10cm_2m_10_bis/10cm_2m_10_bw2'
]

longueur = 2500*115/1875
bas = -100*115/1875
haut = 100*115/1875
fig, axs = plt.subplots(4, 4, figsize=(20,15))
axs = axs.flatten()
count = 0
for seq_idx, folder in enumerate(sequence_folders):
    all_curve = []
    count+=1
    tif_files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.tif')])

    for filename in tif_files:
        input_path = os.path.join(folder, filename)
        img_gray = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        height, width = img_gray.shape
        step = 5
        points_sediment_water = []

        for x in range(0, width, step):
            column = img_gray[:, x].astype(np.float32)
            column = cv2.GaussianBlur(column[:, np.newaxis], (11, 1), 0).squeeze()
            diff = np.diff(column)

            air_to_water_idx = len(diff[1:-1]) - np.argmax(diff[1:-1][::-1]) + 1
            water_to_sediment_idx = len(diff[1:-1]) - np.argmin(diff[1:-1][::-1]) + 1

            if 0 < air_to_water_idx < height - 1 and 0 < water_to_sediment_idx < height - 1:
                if air_to_water_idx < water_to_sediment_idx:
                    points_sediment_water.append((x, water_to_sediment_idx))

        all_curve.append(points_sediment_water)

    
    time_indices = [i for i, pts in enumerate(all_curve) if len(pts) > 0]
    #print(time_indices)
    
    ax = axs[seq_idx]

    
    
    if count == 5:
        time_indices= time_indices[5:] 

    if len(time_indices) >= 4:
        first = time_indices[0]
        quarter = time_indices[len(time_indices) // 4]
        three_quarter = time_indices[(3 * len(time_indices)) // 4]
        last = time_indices[-1]

        selected = [first, last]
        labels = ['Début', 'Fin']
        colors = ['blue', 'red']

        x_all = []
        y_all = []

        for idx, label, color in zip(selected, labels, colors):
            points = np.array(all_curve[idx])
            if len(points) > 0:
                if count != 15 and count != 1 and count != 2 and count !=3 and count !=4 and count !=5 :
                    points = points[4:] #enleve 4 points au depart et 5a la fin points = points[4:-5]  (données abérantes)
                if 0<count<5:
                    x, y = (600+points[:, 0]), points[:, 1]
                else:
                    x, y = points[:, 0], points[:, 1]

                x_all.append(x)
                y_all.append(y)
                

        min_len = min(len(x_all[1]), len(x_all[0]), len(y_all[1]),len(y_all[0]))
        x_common = (-(x_all[0][:min_len])+2500)*115/1875  
        delta_y = [-(y_all[1][i] - y_all[0][i])*115/1875 for i in range(min_len)]
        for i in range(0,len(delta_y)):
            if delta_y[i]>100*115/1875:
                delta_y[i]=100*115/1875
            if delta_y[i]<-100*115/1875:
                delta_y[i]=-100*115/1875


        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        ax.plot(x_common, delta_y, label=label, color="black") #"Δy (fin - début)"

        #ax.invert_yaxis()
        #ax.invert_xaxis()
        ax.set_xticks(np.linspace(0, longueur, 5))  
        ax.set_yticks(np.linspace(bas, haut, 5))
        ax.tick_params(labelsize=12, length=2)  
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
        ax.legend(fontsize=12)
    else:
        ax.axis('off')

col_titles = ["4cm", "7cm", "10cm", "10cm bis"]
for col_idx, title in enumerate(col_titles):
    # On prend l'axe de la première ligne pour chaque colonne (indices 0,1,2,3)
    ax = axs[col_idx]
    ax.text(0.5, 1.15, title, ha='center', va='bottom', fontsize=25, fontweight='bold', transform=ax.transAxes)

# Titres de ligne : distances 1.5m / 2m
row_labels = ["1.5m", "2m", "1.5m", "2m"]
for row_idx, label in enumerate(row_labels):
    # On prend l'axe de la première colonne de chaque ligne (0, 4, 8, 12)
    ax = axs[row_idx * 4]
    fig.text(0.10, ax.get_position().y0 + ax.get_position().height / 2,
             label, ha='center', va='center', fontsize=25, fontweight='bold', rotation='vertical')

#print(all_curve[0])
plt.tight_layout()

# Ajuste les marges pour laisser plus de place à gauche
plt.subplots_adjust(left=0.15, right=0.97, top=0.95, bottom=0.05, hspace=0.3)

# Ajoute les titres verticaux plus visibles et centrés
fig.text(0.04, 0.72, "Pentes fortes", fontsize=25, weight='bold', color='darkred', va='center', rotation='vertical')
fig.text(0.04, 0.28, "Pentes douces", fontsize=25, weight='bold', color='blue', va='center', rotation='vertical')

# Ajoute un fond coloré plus visible mais toujours léger
from matplotlib.patches import Rectangle
fig.patches.extend([
    Rectangle((0.15, 0.52), 0.82, 0.43, transform=fig.transFigure,
              color='red', alpha=0.08, zorder=0, linewidth=2, edgecolor='red'),
    Rectangle((0.15, 0.05), 0.82, 0.42, transform=fig.transFigure,
              color='blue', alpha=0.08, zorder=0, linewidth=2, edgecolor='blue')
])

plt.savefig("test2.png")
plt.show()


#########################################
#pente unique
#########################################
"""import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# === PARAMÈTRES GÉNÉRAUX ===
folder = '4cm_1.5m_10/4cm_1.5m_10_bw'  # À adapter
#folder = '10cm_2m_5/10cm_2m_5_bw'
longueur = 2500 * 115 / 1875
bas = -120 * 115 / 1875
haut = 120 * 115 / 1875
step = 5

# === EXTRACTION DES COURBES ===
all_curve = []
tif_files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.tif')])

for filename in tif_files:
    input_path = os.path.join(folder, filename)
    img_gray = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    height, width = img_gray.shape
    points_sediment_water = []

    for x in range(0, width, step):
        column = img_gray[:, x].astype(np.float32)
        column = cv2.GaussianBlur(column[:, np.newaxis], (11, 1), 0).squeeze()
        diff = np.diff(column)

        air_to_water_idx = len(diff[1:-1]) - np.argmax(diff[1:-1][::-1]) + 1
        water_to_sediment_idx = len(diff[1:-1]) - np.argmin(diff[1:-1][::-1]) + 1

        if 0 < air_to_water_idx < height - 1 and 0 < water_to_sediment_idx < height - 1:
            if air_to_water_idx < water_to_sediment_idx:
                points_sediment_water.append((x, water_to_sediment_idx))

    all_curve.append(points_sediment_water)

# === TRAITEMENT ===
time_indices = [i for i, pts in enumerate(all_curve) if len(pts) > 0]

if len(time_indices) >= 2:
    first = time_indices[0]
    last = time_indices[-1]

    x_all = []
    y_all = []

    for idx in [first, last]:
        points = np.array(all_curve[idx])
        if len(points) > 0:
            points = points[4:]
            x, y = points[:, 0], points[:, 1]
            x_all.append(x)
            y_all.append(y)

    min_len = min(len(x_all[0]), len(x_all[1]))
    x_common = (-(x_all[0][:min_len]) + 2500) * 115 / 1875
    delta_y = [-(y_all[1][i] - y_all[0][i]) * 115 / 1875 for i in range(min_len)]
    delta_y = np.clip(delta_y, bas, haut)

    # === FIGURE POSTER MINIMALISTE AVEC FOND LIMITÉ À L’AXE ===
    fig, ax = plt.subplots(figsize=(8, 4))#, facecolor="none"

    # Ajouter fond dans les limites des axes (zone de traçage)
    ax.add_patch(
        Rectangle((0, bas), longueur, 2*haut ,  #haut - bas
                  color='blue', alpha=0.08, zorder=0, edgecolor='none', linewidth=1.5)
    )

    ax.plot(x_common, delta_y, color='black', linewidth=2.5)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)

    ax.set_xlabel("Position horizontale (cm)", fontsize=18)
    ax.set_ylabel("Évolution verticale (cm)", fontsize=18) #, labelpad=10
    ax.set_xticks(np.linspace(0, longueur, 5))
    ax.set_yticks(np.linspace(bas, haut, 5))
    ax.tick_params(labelsize=14)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xlim(0, longueur)
    ax.set_ylim(bas, haut)
    #ax.invert_xaxis()

    plt.tight_layout()
    plt.savefig("4cm_2m_5_difference.png") #, dpi=300, bbox_inches='tight', facecolor='none'
    #plt.savefig("10cm_1.5m_10_difference.png")
    plt.show()

else:
    print("Pas assez de données valides pour générer une courbe.")"""
