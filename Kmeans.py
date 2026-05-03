import numpy as np

from utils import colors, get_color_prob


def imagen_a_puntos_rgb(imagen):
    """
    Convierte una imagen (alto, ancho, 3) en una lista de filas RGB.
    Cada fila es un píxel: [R, G, B] con valores entre 0 y 255.
    """
    alto, ancho = imagen.shape[0], imagen.shape[1]
    puntos = imagen.reshape(alto * ancho, 3)
    return puntos.astype(np.float64)


class Kmeans:
    """
    K-means en el espacio RGB (cada píxel es un punto de 3 dimensiones).
    Sirve para encontrar los colores más repetidos (centroides).
    """

    def __init__(self, X, K=3):
        # X: matriz (número_de_puntos, 3) con los colores RGB
        self.X = np.asarray(X, dtype=np.float64)
        self.K = int(K)
        self.n_puntos = self.X.shape[0]

        # Se rellenan al llamar a fit()
        self.centroids = None
        self.labels = None

    def _inicializar_centroides(self):
        """
        Elige K píxeles distintos del conjunto como centroides iniciales (aleatorio simple).
        """
        indices = np.random.choice(self.n_puntos, size=self.K, replace=False)
        self.centroids = self.X[indices].copy()

    def fit(self, max_iter=100, semilla=42):
        """
        Repite asignar cada punto al centroide más cercano y mover los centroides a la media
        de sus puntos, hasta que casi no cambien o se llegue a max_iter.
        """
        np.random.seed(semilla)
        self._inicializar_centroides()
        self.labels = np.zeros(self.n_puntos, dtype=int)

        for _ in range(max_iter):
            centroides_viejos = self.centroids.copy()

            # 1) Cada punto va al centroide más cercano (usamos distancia², el mínimo es el mismo que con distancia)
            diff = self.X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]
            dist2 = np.sum(diff * diff, axis=2)
            self.labels = np.argmin(dist2, axis=1)

            # 2) Cada centroide se mueve al centro (media) de sus puntos
            for k in range(self.K):
                mascara = self.labels == k
                if np.sum(mascara) == 0:
                    # Clúster vacío: ponemos el centroide en un punto al azar
                    r = np.random.randint(0, self.n_puntos)
                    self.centroids[k] = self.X[r]
                else:
                    self.centroids[k] = np.mean(self.X[mascara], axis=0)

            # Si casi no se mueven los centroides, paramos antes
            cambio = np.sqrt(np.sum((self.centroids - centroides_viejos) ** 2))
            if cambio < 1e-4:
                break

        return self

    def withinClassDistance(self):
        """
        Suma, para cada clúster, las distancias al cuadrado de cada punto a su centroide.
        Cuanto más bajo, más compactos están los grupos (WCD del enunciado).
        """
        diff = self.X - self.centroids[self.labels]
        return float(np.sum(diff * diff))

    def find_bestK(self, max_K=8):
        """
        Prueba K=2,3,... y devuelve el K del criterio del PDF:
        el primer k tal que mejorar pasando a k+1 baja el WCD menos de un 20%.
        """
        if max_K < 3:
            return 2

        wcd_por_k = []
        for K in range(2, max_K + 1):
            modelo = Kmeans(self.X, K)
            modelo.fit(semilla=42)
            wcd_por_k.append(modelo.withinClassDistance())

        for i in range(len(wcd_por_k) - 1):
            K_actual = 2 + i
            wcd_k = wcd_por_k[i]
            wcd_siguiente = wcd_por_k[i + 1]
            if wcd_k <= 0:
                continue
            mejora = (wcd_k - wcd_siguiente) / wcd_k
            if mejora < 0.20:
                return K_actual

        return max_K

    def get_color(self):
        """
        Para cada centroide RGB, pide a get_color_prob el vector de 11 probabilidades
        y devuelve el nombre del color con más probabilidad (usa utils.colors).
        """
        if self.centroids is None:
            raise RuntimeError("Primero hay que llamar a fit().")

        nombres = []
        for k in range(self.K):
            rgb = np.clip(self.centroids[k], 0, 255).astype(np.uint8)
            # get_color_prob espera una imagen (filas, columnas, 3)
            mini_imagen = rgb.reshape(1, 1, 3)
            probs = get_color_prob(mini_imagen)
            probs = np.asarray(probs).reshape(-1)
            indice = int(np.argmax(probs))
            nombres.append(str(colors[indice]))
        return nombres
