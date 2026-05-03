import numpy as np
from utils_data import visualize_retrieval
from my_labeling import AYUDA, buscar, precision_busqueda, pprint_hits, _pruebas_automaticas

TODAS = {"", "todas", "todo", "todos", "all"}


def num(s, default):
    s = str(s).strip().lower()
    if s in TODAS:
        return default
    try:
        return int(s)
    except ValueError:
        return default


def si(s):
    return str(s).strip().lower() in ("", "s", "si", "sí", "y", "yes")


def main():
    while True:
        op = input("\n== Etiquetador ==\n1 Buscar  2 Tests  3 Precisió  4 Ajuda  0 Sortir\n> ").strip()
        try:
            if op == "0":
                print("Adéu.")
                return
            if op == "1":
                q = input(f"Consulta ({AYUDA}): ").strip()
                if not q:
                    continue
                idx, imgs, _, (pr, co) = buscar(q, limite=num(input("Màx fotos test [totes]: "), None))
                print(f"Tipus={pr}, color={co}, {len(idx)} encerts")
                pprint_hits(idx)
                if idx and si(input("Veure graella? [S/n]: ")):
                    m = max(1, min(num(input("Miniatures [24]: "), 24), len(idx)))
                    sel = idx[:m]
                    visualize_retrieval(np.stack([imgs[i] for i in sel]), m,
                                        info=[f"#{k} i={j}" for k, j in enumerate(sel, 1)],
                                        title=f"{q!r} · {pr}|{co} ({len(idx)} total)")
            elif op == "2":
                _pruebas_automaticas()
            elif op == "3":
                q = input("Consulta: ").strip()
                if q:
                    r = precision_busqueda(q, limite=num(input("Màx fotos [200]: "), 200), salto=3)
                    print("Frase invàlida (cal color+tipus EN)." if r is None
                          else f"Tipus={r[2]}, precisió={round(r[0], 4)}, candidats={r[1]}")
            elif op == "4":
                print(AYUDA)
            else:
                print("0–4")
        except Exception as e:
            print("Error:", e)


if __name__ == "__main__":
    main()
