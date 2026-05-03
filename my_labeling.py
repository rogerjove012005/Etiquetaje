"""
Parte 3: buscador combinado por color y forma (consultas tipo "Pink dress", "black sandals").
Usa KNN para la forma y K-means + get_color para los colores dominantes de cada imagen.
"""

import numpy as np

from KNN import KNN, read_dataset
from Kmeans import Kmeans
from utils import colors as NOMBRES_COLORES

# Nombres de clase exactamente como en images/gt.json (el dataset tiene 9 tipos; el PDF citaba 8 sin "Heels")
FORMAS_VALIDAS = [
    "Dresses",
    "Shirts",
    "Flip Flops",
    "Shorts",
    "Jeans",
    "Socks",
    "Sandals",
    "Handbags",
    "Heels",
]

# Frases o palabras en inglés (minúsculas) -> clase exacta. Las más largas primero.
_CLAVES_FORMA = [
    ("flip flops", "Flip Flops"),
    ("flip-flops", "Flip Flops"),
    ("sandals", "Sandals"),
    ("handbags", "Handbags"),
    ("handbag", "Handbags"),
    ("dresses", "Dresses"),
    ("dress", "Dresses"),
    ("shirts", "Shirts"),
    ("shirt", "Shirts"),
    ("shorts", "Shorts"),
    ("jeans", "Jeans"),
    ("socks", "Socks"),
    ("heels", "Heels"),
]


def parsear_consulta(texto):
    """
    A partir de una frase sencilla en inglés, obtiene (forma, color).
    Devuelve None si no reconoce forma o color.
    Ejemplos: "Pink dress" -> ("Dresses", "Pink"), "black sandals" -> ("Sandals", "Black")
    """
    if not texto or not str(texto).strip():
        return None

    t = texto.lower().strip()

    forma = None
    for clave, etiqueta in _CLAVES_FORMA:
        if clave in t:
            forma = etiqueta
            break

    color = None
    for nombre in NOMBRES_COLORES:
        if nombre.lower() in t:
            color = str(nombre)
            break

    if forma is None or color is None:
        return None
    return forma, color


def _puntos_rgb_muestreados(imagen, paso=2):
    """
    Toma un submuestreo de píxeles (cada 'paso' filas/columnas) para ir más rápido en el K-means.
    """
    trozo = imagen[::paso, ::paso, :]
    n = trozo.shape[0] * trozo.shape[1]
    return trozo.reshape(n, 3).astype(np.float64)


def colores_predominantes(imagen, max_K=8, paso_muestreo=2):
    """
    Devuelve la lista de nombres de color (uno por centroide) tras elegir K con find_bestK y hacer fit.
    """
    X = _puntos_rgb_muestreados(imagen, paso=paso_muestreo)
    if X.shape[0] < 3:
        X = imagen.reshape(-1, 3).astype(np.float64)

    buscador_k = Kmeans(X)
    k = buscador_k.find_bestK(max_K=max_K)
    k = max(2, min(k, X.shape[0]))
    modelo = Kmeans(X, K=k)
    modelo.fit(semilla=42)
    return modelo.get_color()


def etiquetar_imagen(imagen, knn, max_K=8, paso_muestreo=2):
    """
    Etiqueta una sola imagen: (forma_predicha_knn, lista_colores_kmeans).
    """
    forma = knn.predict(np.array([imagen]))[0]
    cols = colores_predominantes(imagen, max_K=max_K, paso_muestreo=paso_muestreo)
    return forma, cols


def buscar(consulta, k_vecinos=5, max_K=8, paso_muestreo=2, conjunto="test", limite=None):
    """
    Busca imágenes que coincidan con la forma y que tengan el color pedido entre los colores de centroides.

    conjunto: "test" o "train"
    limite: si es un entero, solo revisa las primeras N imágenes (útil para pruebas rápidas).

    Devuelve: lista de índices, array de imágenes del conjunto elegido, knn entrenado, (forma, color) parseados
    """
    parsed = parsear_consulta(consulta)
    if parsed is None:
        raise ValueError(
            "No se pudo entender la consulta. Usa inglés con color y prenda, por ejemplo: 'Pink dress', 'Black sandals'."
        )
    forma_objetivo, color_objetivo = parsed

    train_imgs, train_cls, _, test_imgs, _, _ = read_dataset()
    knn = KNN(train_imgs, train_cls, k=k_vecinos)

    if conjunto == "train":
        imgs = train_imgs
    else:
        imgs = test_imgs

    n = len(imgs)
    if limite is not None:
        n = min(n, int(limite))

    indices = []
    for i in range(n):
        forma, cols = etiquetar_imagen(imgs[i], knn, max_K=max_K, paso_muestreo=paso_muestreo)
        if forma != forma_objetivo:
            continue
        if color_objetivo not in cols:
            continue
        indices.append(i)

    return indices, imgs, knn, (forma_objetivo, color_objetivo)


def precision_busqueda(consulta, k_vecinos=5, max_K=8, paso_muestreo=2, limite=200):
    """
    Sobre las primeras 'limite' imágenes de test: entre las que nuestro sistema devuelve como aciertos,
    cuenta cuántas tienen realmente esa forma en la verdad ground-truth (como referencia rápida).
    """
    train_imgs, train_cls, _, test_imgs, test_cls, _ = read_dataset()
    parsed = parsear_consulta(consulta)
    if parsed is None:
        return None
    forma_q, color_q = parsed

    indices, _, _, _ = buscar(
        consulta,
        k_vecinos=k_vecinos,
        max_K=max_K,
        paso_muestreo=paso_muestreo,
        conjunto="test",
        limite=limite,
    )

    if len(indices) == 0:
        return 0.0, 0, forma_q

    bien = sum(1 for i in indices if test_cls[i] == forma_q)
    return bien / len(indices), len(indices), forma_q


def _comprobar():
    """Comprobaciones rápidas sin cargar todas las imágenes dos veces."""
    assert parsear_consulta("Pink dress") == ("Dresses", "Pink")
    assert parsear_consulta("black sandals") == ("Sandals", "Black")
    assert parsear_consulta("red shirt") == ("Shirts", "Red")
    assert parsear_consulta("flip flops") is None  # falta color
    print("parsear_consulta: OK")

    train_imgs, train_cls, _, test_imgs, test_cls, _ = read_dataset()
    knn = KNN(train_imgs, train_cls, k=3)
    # Una imagen debe dar forma coherente con KNN (etiqueta predicha es string)
    f, c = etiquetar_imagen(test_imgs[0], knn, max_K=6, paso_muestreo=3)
    assert str(f) in FORMAS_VALIDAS
    assert len(c) >= 1
    print("etiquetar_imagen (primera test): forma =", f, " colores =", c)

    idx, imgs, _, p = buscar("White shirts", limite=80, paso_muestreo=3, k_vecinos=5)
    print("buscar('White shirts') en primeras 80 test ->", len(idx), "resultados", "| parseado:", p)
    assert isinstance(idx, list)

    prec, nres, _ = precision_busqueda("Black sandals", limite=120, paso_muestreo=3)
    print("precision_busqueda('Black sandals', 120 test):", round(prec, 3), "sobre", nres, "candidatos")

    print("\nTodas las comprobaciones básicas pasaron.")


if __name__ == "__main__":
    _comprobar()
