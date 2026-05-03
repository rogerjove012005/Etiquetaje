"""
Parte 3: buscador por texto (ingles).
KNN -> tipo de prenda | K-means + get_color -> colores fuertes en la foto.
"""

import sys

import numpy as np

from KNN import KNN, read_dataset
from Kmeans import Kmeans
from utils import colors as LISTA_COLORES

# Etiquetas reales del JSON (hay que coincidir exacto al comparar con el KNN)
TIPOS_DE_PRENDA = [
    "Dresses", "Shirts", "Flip Flops", "Shorts", "Jeans",
    "Socks", "Sandals", "Handbags", "Heels",
]

# Texto usuario (minusculas) -> etiqueta del dataset. Largas primero: "flip flops" antes que "sandals".
PALABRA_A_PRENDA = [
    ("flip flops", "Flip Flops"), ("flip-flops", "Flip Flops"),
    ("sandals", "Sandals"), ("handbags", "Handbags"), ("handbag", "Handbags"),
    ("dresses", "Dresses"), ("dress", "Dresses"),
    ("shirts", "Shirts"), ("shirt", "Shirts"),
    ("shorts", "Shorts"), ("jeans", "Jeans"), ("socks", "Socks"),
    ("heels", "Heels"),
]


def parsear_consulta(texto):
    # Convierte frase tipo "Pink dress" en (prenda, color). None si falta uno de los dos.
    if texto is None:
        return None
    texto = str(texto).strip()
    if texto == "":
        return None

    t = texto.lower()

    prenda = None
    for palabra, nombre in PALABRA_A_PRENDA:
        if palabra in t:
            prenda = nombre
            break

    color = None
    for nombre_color in LISTA_COLORES:
        if nombre_color.lower() in t:
            color = str(nombre_color)
            break

    if prenda is None or color is None:
        return None
    return prenda, color


def muestrear_pixeles_rgb(imagen, salto=2):
    # Menos pixeles = K-means mas rapido; salto 2 = uno de cada dos filas y columnas.
    trozo = imagen[::salto, ::salto, :]
    return trozo.reshape(-1, 3).astype(np.float64)


def colores_predominantes(imagen, max_K=8, salto=2):
    # K optimo (find_bestK) + K-means + nombre de color por centroide (get_color).
    X = muestrear_pixeles_rgb(imagen, salto=salto)
    if X.shape[0] < 3:
        X = imagen.reshape(-1, 3).astype(np.float64)

    k = Kmeans(X).find_bestK(max_K=max_K)
    k = max(2, min(k, X.shape[0]))

    modelo = Kmeans(X, K=k)
    modelo.fit(semilla=42)
    return modelo.get_color()


def etiquetar_imagen(imagen, knn, max_K=8, salto=2):
    # Una foto: clase KNN + lista de colores de los centroides.
    prenda = knn.predict(np.array([imagen]))[0]
    colores = colores_predominantes(imagen, max_K=max_K, salto=salto)
    return prenda, colores


def buscar(consulta, k_vecinos=5, max_K=8, salto=2, conjunto="test", limite=None):
    # Recorre imagenes; guarda indice si la prenda KNN coincide y el color buscado esta en la lista K-means.
    par = parsear_consulta(consulta)
    if par is None:
        raise ValueError(
            'Frase no valida. Ejemplo: "Pink dress" o "Black sandals" (ingles, prenda + color).'
        )
    prenda_pedida, color_pedido = par

    train_imgs, train_cls, _, test_imgs, _, _ = read_dataset()
    knn = KNN(train_imgs, train_cls, k=k_vecinos)

    imagenes = train_imgs if conjunto == "train" else test_imgs
    n = len(imagenes)
    if limite is not None:
        n = min(n, int(limite))

    indices = []
    for i in range(n):
        prenda, lista_colores = etiquetar_imagen(imagenes[i], knn, max_K=max_K, salto=salto)
        if prenda != prenda_pedida:
            continue
        if color_pedido not in lista_colores:
            continue
        indices.append(i)

    return indices, imagenes, knn, (prenda_pedida, color_pedido)


def precision_busqueda(consulta, k_vecinos=5, max_K=8, salto=2, limite=200):
    # De lo que devuelve buscar: fraccion donde la prenda real (JSON) coincide con la pedida.
    _, _, _, test_imgs, test_cls, _ = read_dataset()
    par = parsear_consulta(consulta)
    if par is None:
        return None
    prenda_ok, _ = par

    indices, _, _, _ = buscar(
        consulta,
        k_vecinos=k_vecinos,
        max_K=max_K,
        salto=salto,
        conjunto="test",
        limite=limite,
    )
    if len(indices) == 0:
        return 0.0, 0, prenda_ok

    bien = 0
    for i in indices:
        if test_cls[i] == prenda_ok:
            bien += 1
    return bien / len(indices), len(indices), prenda_ok


def _pruebas_automaticas():
    assert parsear_consulta("Pink dress") == ("Dresses", "Pink")
    assert parsear_consulta("black sandals") == ("Sandals", "Black")
    assert parsear_consulta("red shirt") == ("Shirts", "Red")
    assert parsear_consulta("flip flops") is None
    print("[OK] parsear_consulta")

    train_imgs, train_cls, _, test_imgs, _, _ = read_dataset()
    knn = KNN(train_imgs, train_cls, k=3)
    prenda, cols = etiquetar_imagen(test_imgs[0], knn, max_K=6, salto=3)
    assert str(prenda) in TIPOS_DE_PRENDA and len(cols) >= 1
    print("[OK] etiquetar_imagen:", prenda, cols)

    idx, _, _, _ = buscar("White shirts", limite=80, salto=3, k_vecinos=5)
    print("[OK] buscar White shirts (80 test):", len(idx), "resultados")

    p, n, _ = precision_busqueda("Black sandals", limite=120, salto=3)
    print("[OK] precision Black sandals:", round(p, 3), "en", n, "candidatos")
    print("\nPruebas automaticas OK.")


AYUDA = """
========== Probar desde terminal ==========
  cd ruta/al/Proyecto

  python3 my_labeling.py              -> solo muestra esta ayuda
  python3 my_labeling.py --test       -> pruebas rapidas
  python3 my_labeling.py "Pink dress" -> busqueda (test completo = lento)
  python3 my_labeling.py "Blue jeans" --limite 100  -> mas rapido

Colores: Red Orange Brown Yellow Green Blue Purple Pink Black Grey White
Prendas: dress shirt jeans shorts socks sandals flip flops handbag heels
===========================================
"""


def _parse_args_busqueda(argv):
    # Separa "frase de busqueda" del flag opcional --limite N
    limite = None
    palabras = []
    i = 0
    while i < len(argv):
        if argv[i] == "--limite" and i + 1 < len(argv):
            limite = int(argv[i + 1])
            i += 2
        else:
            palabras.append(argv[i])
            i += 1
    return " ".join(palabras).strip(), limite


if __name__ == "__main__":
    args = sys.argv[1:]

    if len(args) == 0:
        print(AYUDA)
        print("Tip: python3 my_labeling.py --test")
        sys.exit(0)

    if args[0] in ("--test",):
        _pruebas_automaticas()
        sys.exit(0)

    if args[0] in ("--help", "-h"):
        print(AYUDA)
        sys.exit(0)

    consulta, limite = _parse_args_busqueda(args)
    if consulta == "":
        print('Escribe una consulta, ej: python3 my_labeling.py "Red shirt"')
        sys.exit(1)

    print("Consulta:", repr(consulta))
    if limite is not None:
        print("Limite:", limite, "primeras imagenes de test")
    else:
        print("Aviso: sin --limite se revisa todo el test (tarda).")

    idx, _, _, (prenda, color) = buscar(consulta, limite=limite)
    print("Filtro: prenda =", prenda, "| color en centroides =", color)
    print("Encontradas:", len(idx), "fotos")
    for k, j in enumerate(idx[:15], start=1):
        print(" ", k, "indice test:", j)
    if len(idx) > 15:
        print(" ...", len(idx) - 15, "mas")
