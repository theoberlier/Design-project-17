
###############################################################################
#BON GRAPH 4*4 evolution pentes
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

point_coo = [[1875,200],[1875,175],[1875,200],[1875,200],
             [1875,120],[1875,120],[1950,120],[1875,120],
             [1875,240],[1875,240],[1875,240],[1875,270],
             [1875,175],[1875,160],[1875,160],[1875,200]]

max_height = 0
for folder in sequence_folders:
    tif_files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.tif')])
    if tif_files:
        sample_img = cv2.imread(os.path.join(folder, tif_files[0]), cv2.IMREAD_GRAYSCALE)
        if sample_img is not None:
            max_height = max(max_height, sample_img.shape[0])
max_height = 637.5*115/1875
longueur = 2500*115/1875

fig, axs = plt.subplots(4, 4, figsize=(20, 15))
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

    # Sélectionner les indices temporels valides
    time_indices = [i for i, pts in enumerate(all_curve) if len(pts) > 0]
    #print(time_indices)
    
    ax = axs[seq_idx]

    ax.set_ylim(0,max_height )  # Échelle verticale unifiée
    
    if count == 5:
        time_indices= time_indices[5:] #enleve des images au debut time_indices= time_indices[5:]
    #if count == 12:
        #time_indices= time_indices[200:] # on retire cette image car manque de donnée utilisable
    #if count == 4 :
    #    time_indices= time_indices[:-200]

    if len(time_indices) >= 4:
        first = time_indices[0]
        quarter = time_indices[len(time_indices) // 3]
        three_quarter = time_indices[(2 * len(time_indices)) // 3]
        last = time_indices[-1]

        selected = [first, quarter, three_quarter, last]
        labels = ['Début', '1/3', '2/3', 'Fin']
        colors = ['blue', 'green', 'orange', 'red']

        for idx, label, color in zip(selected, labels, colors):
            points = np.array(all_curve[idx])
            if len(points) > 0:
                if count != 15 and count != 1 and count != 2 and count !=3 and count !=4 and count !=5 :
                    points = points[4:] #enleve 4 points au depart et 5a la fin points = points[4:-5]
                #if count == 13:
                #    points = points[6:]
                if 0<count<5: 
                    x, y = -(600+points[:, 0])+2500, -(points[:, 1])+637.5
                    x = x*115/1875
                    y = y*115/1875
                else :
                    x, y = -(points[:, 0])+2500, -(points[:, 1])+637.5
                    x = x*115/1875
                    y = y*115/1875
                ax.plot(x, y, label=label, color=color)
                #ax.plot(-point_coo[count-1][0]+2500, -point_coo[count-1][1]+637.5, 'kx', markersize=5, label='_nolegend_')
                ax.plot((-point_coo[count-1][0]+2500)*115/1875, (-point_coo[count-1][1]+637.5)*115/1875, 'kx', markersize=5, label='_nolegend_')                

        #width = 2500*115/1875
        #max_height = max_height*115/1875
        #ax.set_title(f'Séquence {seq_idx + 1}')
        #ax.invert_yaxis()
        #ax.invert_xaxis()
        ax.set_xticks(np.linspace(0, longueur, 5))  # ou une autre valeur selon le visuel souhaité         
        ax.set_yticks(np.linspace(0, max_height, 5))#
        ax.tick_params(labelsize=12, length=2)  # pour que ce soit discret
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
        ax.legend(fontsize=12)
    else:
        #ax.set_title(f'Séquence {seq_idx + 1} (données insuffisantes)')
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
#plt.suptitle("Évolution de l'interface eau-sédiment sur 16 séquences", fontsize=22, y=1.02)

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

#axs[3].set_visible(False)

plt.savefig("test.png")

plt.show()

##########################################################3
#Pente unique 
###########################################################
"""import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# Paramètres de la séquence ciblée
#folder = '4cm_1.5m_10/4cm_1.5m_10_bw'
folder = '7cm_2m_5/7cm_2m_5_bw'
point_ref = [1850, 240]  # rive initiale
point_ref = [1975, 110]

# Calcule les dimensions physiques réelles
max_height = 637.5 * 115 / 1875
longueur = 2500 * 115 / 1875

# Lecture des fichiers
tif_files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.tif')])
all_curve = []

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

# Indices temporels valides
time_indices = [i for i, pts in enumerate(all_curve) if len(pts) > 0]


fig, ax = plt.subplots(figsize=(8, 4))


ax.add_patch(
    Rectangle(
        (0, 0), longueur, max_height,
        color='red', alpha=0.08, zorder=0, linewidth=1.5, edgecolor='blue'
    )
)


if len(time_indices) >= 4:
    selected = [
        time_indices[0],
        time_indices[len(time_indices) // 3],
        time_indices[(2 * len(time_indices)) // 3],
        time_indices[-1]
    ]
    labels = ['Début', '1/3', '2/3', 'Fin']
    colors = ['blue', 'green', 'orange', 'red']

    for idx, label, color in zip(selected, labels, colors):
        points = np.array(all_curve[idx])
        points = points[4:]
        x, y = -(points[:, 0]) + 2500, -(points[:, 1]) + 637.5
        x = x * 115 / 1875 
        y = y * 115 / 1875
        ax.plot(x, y, label=label, color=color)

    # Marqueur de référence
    x_ref = (-point_ref[0] + 2500) * 115 / 1875
    y_ref = (-point_ref[1] + 637.5) * 115 / 1875
    ax.plot(x_ref, y_ref, 'kx', markersize=5, label='_nolegend_')


ax.set_xlim(0, longueur)
ax.set_ylim(0, max_height)
ax.set_xticks(np.linspace(0, longueur, 5))
ax.set_yticks(np.linspace(0, max_height, 5))
ax.tick_params(labelsize=14)
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
ax.set_xlabel("Longueur (cm)", fontsize=18)
ax.set_ylabel("Hauteur (cm)", fontsize=18)
ax.legend(fontsize = 14)

plt.tight_layout()
plt.savefig("10cm_2m_5_plot_limited_bg.png")
#plt.savefig("4cm_1.5m_10_plot_limited_bg.png")
plt.show()"""