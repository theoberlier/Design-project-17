###############################################################
# Courbe pour 4 expériences
###############################################################
"""
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

# === PARAMÈTRES ===
folders = ['4cm_1.5m_5/4cm_1.5m_5_bw','4cm_2m_5/4cm_2m_5_bw','4cm_1.5m_10/4cm_1.5m_10_bw','4cm_2m_10/4cm_2m_10_bw']
#folders= ['7cm_1.5m_5/7cm_1.5m_5_bw','7cm_2m_5/7cm_2m_5_bw','7cm_1.5m_10/7cm_1.5m_10_bw','7cm_2m_10/7cm_2m_10_bw']
#folders = ['10cm_1.5m_5/10cm_1.5m_5_bw','10cm_2m_5/10cm_2m_5_bw','10cm_1.5m_10/10cm_1.5m_10_bw','10cm_2m_10/10cm_2m_10_bw']
#folders =['10cm_1.5m_5/10cm_1.5m_5_bw','10cm_2m_5_bis/10cm_2m_5_bw2','10cm_1.5m_10_bis/10cm_1.5m_10_bw2','10cm_2m_10_bis/10cm_2m_10_bw2']

param = ["1.5m / 1 : 5", "2m / 1 : 5","1.5m / 1 : 10","2m / 1 : 10"]
limit = 0.36
step = 5
delta_t = 30  # seconds
scaling_factor = 115 / 1875  # for y-axis scaling

# === PLOTTING SETUP ===
#fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig, axes = plt.subplots(1, 4, figsize=(28, 5))
axes = axes.flatten()

for idx, folder in enumerate(folders):
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

    # Skip last 5 curves if desired
    #all_curve = all_curve[:-5]

    if len(all_curve) < 2:
        continue  # Not enough data to compute speed

    # === Extract common x positions ===
    x_sets = [set([pt[0] for pt in curve]) for curve in all_curve]
    x_communs = sorted(set.intersection(*x_sets))

    pentes = np.full((len(all_curve), len(x_communs)), np.nan)
    for i, curve in enumerate(all_curve):
        dict_curve = dict(curve)
        for j, x_val in enumerate(x_communs):
            if x_val in dict_curve:
                pentes[i, j] = dict_curve[x_val]

    # Interpolate missing values
    for i in range(len(pentes)):
        if np.any(np.isnan(pentes[i])):
            valid = ~np.isnan(pentes[i])
            pentes[i, ~valid] = np.interp(
                np.array(x_communs)[~valid],
                np.array(x_communs)[valid],
                pentes[i, valid]
            )

    # === Compute speed ===
    temps = np.arange(len(all_curve)) * 0.5  # minutes
    vitesses = []
    temps_vitesses = []

    for i in range(1, len(temps)):
        diff = pentes[i] * scaling_factor - pentes[i - 1] * scaling_factor
        D = np.sqrt(np.sum(diff**2))
        vitesse = D / delta_t
        if vitesse > limit:
            vitesse = limit

        if i > 2:
            if abs(vitesse - vitesses[-1]) > 0.15:
                vitesse = vitesses[-2]

        vitesses.append(vitesse)
        temps_vitesses.append((temps[i] + temps[i - 1]) / 2)


    # === Apply Savitzky-Golay smoothing ===
    if len(vitesses) >= 5:
        window_length = max(5, len(vitesses) // 3)
        if window_length % 2 == 0:
            window_length += 1
        polyorder = 2
        vitesses_lisse = savgol_filter(vitesses, window_length=window_length, polyorder=polyorder)
    else:
        vitesses_lisse = vitesses

    # === Plotting ===
    ax = axes[idx]
    ax.plot(temps_vitesses, vitesses, 'o-', markersize=4, label="Vitesse brute")
    ax.plot(temps_vitesses, vitesses_lisse, 'r--', label="Tendance (Savitzky-Golay)")
    #ax.set_title(f"Dossier {idx+1}")
    ax.set_title(param[idx], fontsize=30)
    ax.set_xlabel("Temps (min)", fontsize=30)
    ax.set_ylabel("Vitesse (cm/s)", fontsize=30)
    ax.grid(True)
    ax.legend(fontsize=18)
    ax.set_xticks(np.linspace(0,50, 5))  
    ax.set_yticks(np.arange(0, limit,0.1))
    ax.tick_params(labelsize=22, length=2)

plt.tight_layout()
#plt.savefig("vitesse_10cm")
#plt.savefig("vitesse_10cm_bis")
#plt.savefig("vitesse_7cm")
plt.savefig("vitesse_4cm")
plt.show()"""

###################################################
#courbe pour une seule expérience 
###################################################
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

# === PARAMÈTRES ===
folder = '7cm_2m_5/7cm_2m_5_bw'
param = "2m / 1 : 5"
limit = 0.36
step = 5
delta_t = 30  # secondes
scaling_factor = 115 / 1875

# === EXTRACTION COURBES ===
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

# all_curve = all_curve[:-5]  # facultatif

# === CONVERSION MATRICE 2D ===
x_sets = [set([pt[0] for pt in curve]) for curve in all_curve]
x_communs = sorted(set.intersection(*x_sets))
pentes = np.full((len(all_curve), len(x_communs)), np.nan)

for i, curve in enumerate(all_curve):
    dict_curve = dict(curve)
    for j, x_val in enumerate(x_communs):
        if x_val in dict_curve:
            pentes[i, j] = dict_curve[x_val]

# Interpolation des NaN
for i in range(len(pentes)):
    if np.any(np.isnan(pentes[i])):
        valid = ~np.isnan(pentes[i])
        pentes[i, ~valid] = np.interp(
            np.array(x_communs)[~valid],
            np.array(x_communs)[valid],
            pentes[i, valid]
        )

# === CALCUL DES VITESSES ===
temps = np.arange(len(all_curve)) * 0.5  # en minutes
vitesses = []
temps_vitesses = []

for i in range(1, len(temps)):
    diff = pentes[i] * scaling_factor - pentes[i - 1] * scaling_factor
    D = np.sqrt(np.sum(diff**2))
    vitesse = D / delta_t
    if vitesse > limit:
        vitesse = limit

    if i > 2:
        if abs(vitesse - vitesses[-1]) > 0.15:
            vitesse = vitesses[-2]

    vitesses.append(vitesse)
    temps_vitesses.append((temps[i] + temps[i - 1]) / 2)

# === LISSAGE SG ===
if len(vitesses) >= 5:
    window_length = max(5, len(vitesses) // 3)
    if window_length % 2 == 0:
        window_length += 1
    vitesses_lisse = savgol_filter(vitesses, window_length=window_length, polyorder=2)
else:
    vitesses_lisse = vitesses

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(temps_vitesses, vitesses, 'o-', markersize=4, label="Vitesse brute")
ax.plot(temps_vitesses, vitesses_lisse, 'r--', label="Tendance (Savitzky-Golay)")
ax.set_title(param, fontsize=20)
ax.set_xlabel("Temps (min)", fontsize=20)
ax.set_ylabel("Vitesse (cm/s)", fontsize=20)
ax.grid(True)
ax.set_xticks(np.linspace(0, 50, 5))
ax.set_yticks(np.arange(0, limit, 0.1))
ax.legend(fontsize=15)

plt.tight_layout()
plt.savefig("vitesse_7cm_2m_5.png", dpi=300)
plt.show()
