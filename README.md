# Etiquetador — pràctica (forma + color + cercador)

Etiquetatge automàtic d’imatges de botiga: **forma** amb **k-NN**, **colors predominants** amb **K-means** + model del curs, i **cerca** per text en anglès (color + peça).

- **Informe (Markdown):** [`Informe.md`](Informe.md)  
- **Informe (PDF):** [`informe_etiquetador.pdf`](informe_etiquetador.pdf)  
- **Repositori:** [github.com/rogerjove012005/Etiquetaje](https://github.com/rogerjove012005/Etiquetaje)

## Requisits

Python 3.9+ (recomanat 3.10+). Instal·lació:

```bash
pip install -r requirements.txt
```

(`numpy`, `Pillow`, `matplotlib`)

## Estructura

| Fitxer | Rol |
|--------|-----|
| `main.py` | **Entrada habitual:** menú (cerca, proves, precisió, ajuda) i opció de veure candidats amb `visualize_retrieval` |
| `my_labeling.py` | Part 3: `parsear_consulta`, `buscar`, `precision_busqueda`; només CLI `python3 my_labeling.py --test` |
| `KNN.py` | Part 1: `read_dataset` (envoltori de `utils_data`) + classe `KNN` (vector en gris, vot majoritari) |
| `Kmeans.py` | Part 2: `imagen_a_puntos_rgb`, `Kmeans` (`fit`, WCD, `find_bestK` 20%, `get_color`) |
| `utils.py` / `utils_data.py` | Material del curs (color, càrrega d’imatges, visualització) |
| `images/` | `train/`, `test/`, `gt.json` |

## Com executar-ho

Des de la carpeta del projecte:

```bash
# Menú interactiu (cerca, graella de resultats, precisió)
python3 main.py

# Proves automàtiques (parseig + etiquetatge + cerca curta)
python3 my_labeling.py --test
```

Exemple de consulta vàlida al menú **1** (anglès, color + peça): `Pink dress`, `Black sandals`, `Blue shorts`.  
Per al límit de fotos de test: **Enter** o `todas` = revisar tot el conjunt (més lent).

## Consultes suportades

- **Colors:** Red, Orange, Brown, Yellow, Green, Blue, Purple, Pink, Black, Grey, White  
- **Peces (paraules clau al text):** dress, shirt, jeans, shorts, socks, sandals, flip flops, handbag, heels, …

## Provar des de Python

```python
from KNN import read_dataset, KNN

train_imgs, train_cls, _, test_imgs, _, _ = read_dataset()
knn = KNN(train_imgs, train_cls, k=5)
pred = knn.predict(test_imgs[:10])
```

```python
from KNN import read_dataset
from Kmeans import Kmeans, imagen_a_puntos_rgb

train_imgs, _, _, _, _, _ = read_dataset()
X = imagen_a_puntos_rgb(train_imgs[0])
km = Kmeans(X, K=3).fit()
print(km.get_color())
```

## Llicència / ús

Codi per a l’assignatura; les imatges i `gt.json` pertanyen al material del curs.
