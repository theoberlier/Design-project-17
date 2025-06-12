from PIL import Image
import os
import shutil


#Nous avions besoin de transformer les .tif que nous redonnait ImageJ et qui contenait toutes les images 
# pour en faire plusieurs contenant qu'une seule image traitable sur Python

def extraire_pages_tif(chemin_fichier):
    nom_base = os.path.splitext(os.path.basename(chemin_fichier))[0]
    dossier_sortie = f"{nom_base}_pages"
    os.makedirs(dossier_sortie, exist_ok=True)

    with Image.open(chemin_fichier) as img:
        for i in range(img.n_frames):
            img.seek(i)
            chemin_image = os.path.join(dossier_sortie, f"{nom_base}_page_{i+1:03}.tif")
            img.save(chemin_image, "TIFF")
            print(f"Page {i+1} enregistrée : {chemin_image}")

    # Créer un fichier ZIP du dossier
    shutil.make_archive(dossier_sortie, 'zip', dossier_sortie)
    print(f"Archive créée : {dossier_sortie}.zip\n")

if __name__ == "__main__":
    # Liste des fichiers à traiter
    fichiers_tif = [
        "4cm_1.5m_10/4cm_1.5m_10_bw.tif",
        "7cm_1.5m_10/7cm_1.5m_10_bw.tif",
        "10cm_1.5m_10/10cm_1.5m_10_bw.tif",
        "10cm_1.5m_10_bis/10cm_1.5m_10_bw2.tif",
        "4cm_2m_10/4cm_2m_10_bw.tif",
        "7cm_2m_10/7cm_2m_10_bw.tif",
        "10cm_2m_10/10cm_2m_10_bw.tif",
        "10cm_2m_10_bis/10cm_2m_10_bw2.tif"
    ]

    for fichier in fichiers_tif:
        if os.path.exists(fichier):
            extraire_pages_tif(fichier)
        else:
            print(f"Fichier introuvable : {fichier}")
