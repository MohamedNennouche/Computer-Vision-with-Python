# Librairies pour la manipulation des fichiers et des variables
import os
import glob
import shutil
import csv

# Pour la manipulation de matrices
import numpy as np

# Pour le split des images aléatoirement 
from sklearn.model_selection import train_test_split

# Pour le prétraitement des images et la mise en place du modèle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


""" 
============================================== FONCTIONS ================================================
"""

def split_data(path_to_data:str, path_to_save_train:str, path_to_save_val:str, split_size:float=0.1) -> None:
    """Fonction qui permet de spliter les images d'entrainements aléatoirement en deux dossiers validation et entrainement

    Args:
        - path_to_data (str): Chemin vers le dossier qui contient les dossiers avec toutes les images (un dossier par classe)
        - path_to_save_train (str): Chemin où sauvegarder les images sélectionnées pour l'entrainement
        - path_to_save_val (str): Chemin où sauvegarder les images sélectionnées pour la validation
        - split_size (float, optional): Pourcentage de l'ensemble de validation sur la totalité des images. Defaults to 0.1.
    """
    folders = os.listdir(path_to_data) # la liste des dossiers disponible au chemin donné
    for folder in folders :
        full_path = os.path.join(path_to_data, folder) # pour avoir le chemin complet en ajoutant le nom des dossiers
        images_paths = glob.glob(os.path.join(full_path, '*.png')) # ca prend tous les fichiers à l'intérieur du dossier et les télécharge (le join il va a chaque fois ajouter le path du dossier et ajoutant le nom du fichier) ca nous retourne une liste d'images
        x_train, x_val = train_test_split(images_paths, test_size=split_size) # split en train et validation

        for x in x_train : 
            path_to_folder = os.path.join(path_to_save_train, folder) # pour recréer les même dossier que dans le dossier de base
            if not os.path.isdir(path_to_folder) : 
                os.makedirs(path_to_folder) # si il n'existe pas il le crée
            shutil.copy(x, path_to_folder)
        
        for x in x_val : 
            path_to_folder = os.path.join(path_to_save_val, folder) # pour recréer les même dossier que dans le dossier de base
            if not os.path.isdir(path_to_folder) : 
                os.makedirs(path_to_folder) # si il n'existe pas il le crée
            shutil.copy(x, path_to_folder)


def order_test_set(path_to_images:str, path_to_csv:str) -> None :
    """Fonction permettant de mettre en forme le dossier de test en classes pour permettre l'évaluation du modèle

    Args:
        - path_to_images (str): Chemin vers le dossier de test
        - path_to_csv (str): Chemin vers le fichier CSV contenant les classes de chaque image de test
    """
    try : 
        with open(path_to_csv, 'r') as csvfile : # On ouvre le fichier d'une façon temporaire
            reader = csv.reader(csvfile, delimiter=',')
            for i, row in enumerate(reader) :
                if i == 0 : 
                    continue # pour ne pas prendre en compte la première ligne
                img_name = row[-1].replace('Test/','') # prendre la dernière colonne qui est le nom de l'image en enlevant le 'Test/' au début de chaque nom d'image
                label = row[-2]

                path_to_folder = os.path.join(path_to_images,label) # on crée un dossier avec le nom du label comme le dataset d'entrainement
                if not os.path.isdir(path_to_folder) :
                    os.makedirs(path_to_folder)

                img_full_path = os.path.join(path_to_images, img_name)

                shutil.move(img_full_path, path_to_folder) # on va déplacer et pas copier
    except : 
        print("On ne peut pas ouvrir le fichier csv") 


def create_generators(batch_size:int, train_data_path:str, val_data_path:str, test_data_path:str) :
    """Fonction permettant de mettre en place les générateurs pour les images d'entraînement, de validation et de test

    Args:
        - batch_size (int): Taille du batch d'images qu'on veut utiliser
        - train_data_path (str): Chemin vers les images d'entraînements
        - val_data_path (str): Chemin vers les images de validation
        - test_data_path (str): Chemin vers les images de test

    Returns:
        - ImageDataGenerator: Les 3 générateurs d'entrainement, de validation et de test
    """
    preprocessor = ImageDataGenerator(
        rescale = 1/255. # pour assurer une division flottante
    )

    # !très important pour prendre des données en prenant chaque sous dossier comme classe à part entière
    train_generator = preprocessor.flow_from_directory(
        train_data_path,
        class_mode="categorical",
        target_size=(60,60), #resize all images
        color_mode = 'rgb', # type d'images
        shuffle = True, # tres important
        batch_size=batch_size
    )

    val_generator = preprocessor.flow_from_directory(
        val_data_path,
        class_mode="categorical",
        target_size=(60,60), #resize all images
        color_mode = 'rgb', # type d'images
        shuffle = False,
        batch_size=batch_size
    )

    test_generator = preprocessor.flow_from_directory(
        test_data_path,
        class_mode="categorical",
        target_size=(60,60), #resize all images
        color_mode = 'rgb', # type d'images
        shuffle = False,
        batch_size=batch_size
    )

    return train_generator, val_generator, test_generator


def predict_with_model(model, img_path:str) -> int : 
    """Fonction pour utiliser notre modèle sur une image

    Args:
        model (modele Tensorflow): Modèle pré-entrainé chargé en mémoire
        img_path (str): Chemin vers l'image sur laquelle on veut faire le test

    Returns:
        int: Prédiction de la classe par la modèle
    """
    image = tf.io.read_file(img_path) # On lit l'image
    image = tf.image.decode_png(image, channels=3) # On la décode
    image = tf.image.convert_image_dtype(image, dtype=tf.float32) # Convertir les entiers en float, cela permet de rescaler nos images : A VOIR !
    image = tf.image.resize(image , [60,60]) # resize les images et on a alors de la forme (60,60,3)
    image = tf.expand_dims(image, axis=0) # pour avoir au final (1,60,60,3) pour l'adapter à notre modèle (voir summary du modèle il attend un tel format)

    prediction = model.predict(image) # peut etre une décision ou un ensemble de probabilités (pour chaque classe)
    prediction = np.argmax(prediction) # Pour avoir l'indexe de la meilleure probabilité et par conséquent le label
    return prediction