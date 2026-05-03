import sys
from KNN import KNN, read_dataset
from Kmeans import Kmeans, imagen_a_puntos_rgb
from utils import colors as COLORES

AYUDA = 'Frase EN amb COLOR i TIPUS, p.ex. "Pink dress" o "Black sandals".'
TIPOS = ["Dresses", "Shirts", "Flip Flops", "Shorts", "Jeans", "Socks", "Sandals", "Handbags", "Heels"]
ALIAS = {
    "flip flops": "Flip Flops", "flip-flops": "Flip Flops",
    "sandals": "Sandals", "handbags": "Handbags", "handbag": "Handbags",
    "dresses": "Dresses", "dress": "Dresses",
    "shirts": "Shirts", "shirt": "Shirts",
    "shorts": "Shorts", "jeans": "Jeans", "socks": "Socks", "heels": "Heels",
}


def parsear_consulta(t):
    t = str(t or "").strip().lower()
    if not t:
        return None
    pr = next((v for k, v in ALIAS.items() if k in t), None)
    co = next((str(c) for c in COLORES if c.lower() in t), None)
    return (pr, co) if pr and co else None


def pprint_hits(ix, lim=15):
    for k, j in enumerate(ix[:lim], 1):
        print(f" {k} idx={j}")
    if len(ix) > lim:
        print(f" …+{len(ix)-lim}")


def colores_predominantes(im, max_K=8, salto=2):
    X = imagen_a_puntos_rgb(im[::salto, ::salto])
    if len(X) < 3:
        X = imagen_a_puntos_rgb(im)
    k = max(2, min(Kmeans(X).find_bestK(max_K), len(X)))
    return Kmeans(X, k).fit().get_color()


def etiquetar_imagen(im, knn, max_K=8, salto=2):
    return knn.predict([im])[0], colores_predominantes(im, max_K, salto)


def buscar(consulta, k_vecinos=5, max_K=8, salto=2, limite=None):
    par = parsear_consulta(consulta)
    if par is None:
        raise ValueError(AYUDA)
    pd, cp = par
    tr, ty, _, te, _, _ = read_dataset()
    knn = KNN(tr, ty, k=k_vecinos)
    n = len(te) if limite is None else min(len(te), int(limite))
    ix = []
    for i in range(n):
        pr, cols = etiquetar_imagen(te[i], knn, max_K, salto)
        if pr == pd and cp in cols:
            ix.append(i)
    return ix, te, knn, (pd, cp)


def precision_busqueda(consulta, k_vecinos=5, max_K=8, salto=2, limite=200):
    par = parsear_consulta(consulta)
    if par is None:
        return None
    pd, _ = par
    _, _, _, _, ty, _ = read_dataset()
    ix, *_ = buscar(consulta, k_vecinos, max_K, salto, limite)
    if not ix:
        return 0.0, 0, pd
    return sum(1 for i in ix if ty[i] == pd) / len(ix), len(ix), pd


def _pruebas_automaticas():
    assert parsear_consulta("Pink dress") == ("Dresses", "Pink")
    assert parsear_consulta("black sandals") == ("Sandals", "Black")
    assert parsear_consulta("flip flops") is None
    print("[OK] parsear_consulta")
    tr, ty, _, te, _, _ = read_dataset()
    pr, cols = etiquetar_imagen(te[0], KNN(tr, ty, k=3), max_K=6, salto=3)
    assert pr in TIPOS and cols
    print("[OK] etiquetar_imagen:", pr, cols)
    ix, *_ = buscar("White shirts", limite=80, salto=3)
    print(f"[OK] buscar White shirts (80 test): {len(ix)}")
    p, n, _ = precision_busqueda("Black sandals", limite=120, salto=3)
    print(f"[OK] precision Black sandals: {round(p, 3)} en {n} candidats")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--test"]:
        _pruebas_automaticas()
    else:
        print(f"{AYUDA}\nÚs: python3 main.py")
