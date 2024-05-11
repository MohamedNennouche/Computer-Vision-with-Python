import cv2 as cv
import numpy as np
import pandas as pd
from skimage.feature import hog 
from sklearn.model_selection import train_test_split
import os
import glob
import seaborn as sns

sns.set_theme(style="darkgrid")


def load_images_from_folder(folder="./Fichiers_seance_4/Signalisation/Avertissement") -> list : 
    """Fonction pour charger les images pour leurs utilisation

    Args:
        folder (path): Chemin vers le dossier contenant les images

    Returns:
        List: Liste des images (sous format Numpy) contenu dans le dossier sélectionné
    """
    files = glob.glob(os.path.join(folder, "*.png"))
    images = [cv.cvtColor(cv.imread(file),cv.COLOR_BGR2RGB) for file in files]
    return images

def conv_RGB_YCbCr (listimage:list) -> list :
    """Fonction convertissant la base de couleur de RGB à Y'CbCr 

    Args:
        listimage (list): Liste des images RGB qu'on va traiter

    Returns:
        list: Liste des images qui ont été convertie en Y'CbCr
    """

    channels = [cv.split(listimage[i]) for i in range(len(listimage))]

    y = [0.299*channels[i][0] + 0.587*channels[i][1] + 0.114*channels[i][2] for i in range(len(listimage))]
    Cb = [-0.1687*channels[i][0] - 0.3313*channels[i][1] + 0.5*channels[i][2] + 128 for i in range(len(listimage))]
    Cr = [0.5*channels[i][0] - 0.4187*channels[i][1] - 0.0813*channels[i][2] + 128 for i in range(len(listimage))]

    nouvelles_images = [np.uint8(cv.merge([y[i],Cb[i],Cr[i]])) for i in range(len(listimage))]
    
    return nouvelles_images

def image_resizer (listimage:list) -> None :
    """Fonction pour faire un resizing des images

    Args:
        listimage (list): Liste des images à resizer
    """
    dim = (64, 128)
    for i in range(len(listimage)) : 
        listimage[i] = cv.resize(listimage[i],dim)
    
    return listimage

def hog_list (listimage) :
    """Fonction calculant la liste des HoG d'une liste d'images

    Args:
        listimage (list): Liste des images préalablement traitées

    Returns:
        list, list: Liste des vecteurs caractéristiques du HoG (fv) et la liste des images convertie
    """
    fv = []
    hog_image = []
    for i in range(len(listimage)) :
        a, b = hog(listimage[i], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, channel_axis=-1)
        fv.append(a)
        hog_image.append(b)
    
    return fv,hog_image

def prep_data(listfeatures : list, test_size:float=0.5, random_state:int=42) : 
    """Fonction permettant la préparation des données pour les passer au modèles en les étiquetant et les regroupant

    Args:
        - listfeatures (list): Liste des features qu'on va garder pour le modèle
        - test_size (float): Pourcentage des echantillons pour le test du modèle. Par défaut à 0.5
        - random_state(int): Graine d'initialisation de l'algorithme d'aléatoire pour le split des données. Par défaut à 42

    Returns:
        - fv_train : DataFrame avec tous les échantillons pour l'entrainement du modèle
        - fv_train : DataFrame avec tous les échantillons pour le test du modèle
        - etiq_train : Vecteur Numpy avec les labels pour l'entraînement du modèle
        - etiq_test : Vecteur Numpy avec les labels pour l'évaluation du modèle
    """
    fv = list()
    labels = list()
    i = 0
    for fv_image in listfeatures :
        fv += fv_image
        labels += (i*np.ones(len(fv_image))).astype(np.uint8).tolist() # On crée les labels (0, 1 et 2 dans notre cas)
        i += 1
    fv = pd.DataFrame(fv)
    labels = np.array(labels)

    # Split des données en train et test
    fv_train, fv_test, labels_train, labels_test = train_test_split(fv, labels, test_size=test_size,random_state=random_state) 

    return fv_train, fv_test, labels_train, labels_test