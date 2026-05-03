import numpy as np
from collections import Counter

from utils import rgb2gray
from utils_data import read_dataset as _read_dataset_from_utils


def read_dataset(root_folder="./images/", gt_json="./images/gt.json", w=60, h=80, with_color=True):
    """
    Carga las imágenes de entrenamiento y de test desde la carpeta y el JSON de etiquetas.
    Devuelve: train_imgs, train_clases, train_colores, test_imgs, test_clases, test_colores
    """
    return _read_dataset_from_utils(
        root_folder=root_folder,
        gt_json=gt_json,
        w=w,
        h=h,
        with_color=with_color,
    )


def _imagen_a_vector_gris(imagen):
    """
    Una imagen de ropa la representamos como una fila de números (solo brillo, sin color).
    Así el KNN puede comparar dos imágenes midiendo qué tan lejos están esos vectores.
    """
    # imagen puede ser (alto, ancho, 3) en color o (alto, ancho) ya en gris
    if len(imagen.shape) == 3 and imagen.shape[2] == 3:
        en_gris = rgb2gray(imagen)
    else:
        en_gris = imagen
    # .flatten() pone todos los píxeles en una sola lista larga
    vector = en_gris.flatten().astype(np.float64)
    return vector


class KNN:
    """
    Clasificador "k vecinos más cercanos":
    para cada imagen nueva, mira las k imágenes de entrenamiento más parecidas
    y predice la clase que más veces aparece entre ellas.
    """

    def __init__(self, train_data, train_labels, k=5):
        self.k = k
        self.train_labels = np.array(train_labels)

        # Guardamos cada imagen de entrenamiento ya convertida a un vector de grises
        lista_vectores = []
        for imagen in train_data:
            lista_vectores.append(_imagen_a_vector_gris(imagen))
        self.train_vectores = np.array(lista_vectores)

    def predict(self, test_data):
        predicciones = []

        for imagen_test in test_data:
            vector_test = _imagen_a_vector_gris(imagen_test)

            # Distancia euclídea a cada imagen de entrenamiento:
            # raíz cuadrada de la suma de (diferencia de cada píxel al cuadrado)
            diferencias = self.train_vectores - vector_test
            distancias = np.sqrt(np.sum(diferencias * diferencias, axis=1))

            # Ordenamos de la más cercana a la más lejana y nos quedamos con las k primeras
            indices_ordenados = np.argsort(distancias)
            indices_vecinos = indices_ordenados[: self.k]

            # Las etiquetas de esos vecinos "votan"
            votos = []
            for idx in indices_vecinos:
                votos.append(self.train_labels[idx])

            # Gana la etiqueta que más veces salió
            conteo = Counter(votos)
            etiqueta_ganadora = conteo.most_common(1)[0][0]
            predicciones.append(etiqueta_ganadora)

        return np.array(predicciones, dtype=self.train_labels.dtype)
