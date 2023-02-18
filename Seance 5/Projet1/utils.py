import os
import glob
import shutil
import csv

from sklearn.model_selection import train_test_split


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
        path_to_images (str): Chemin vers le dossier de test
        path_to_csv (str): Chemin vers le fichier CSV contenant les classes de chaque image de test
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