# Etiquetador — pràctica (forma + color + cercador)

Projecte d’**etiquetatge automàtic** d’imatges de botiga: classificació de **forma** amb **k-NN**, **colors predominants** amb **K-means** i **cerca** per text en anglès.

- **Informe acadèmic (català):** veure [`Informe.md`](Informe.md)  
- **Codi remot:** [github.com/rogerjove012005/Etiquetaje](https://github.com/rogerjove012005/Etiquetaje)

## Requisits

- Python 3.9+ (recomanat 3.10+)
- Dependències:

```bash
pip install -r requirements.txt
```

(`numpy`, `Pillow`, `matplotlib`)

## Estructura del repositori

| Fitxer | Descripció |
|--------|------------|
| `KNN.py` | Part 1: lectura de dades (`read_dataset`) i classificador KNN sobre píxels en gris |
| `Kmeans.py` | Part 2: K-means RGB, WCD, `find_bestK` (criteri 20%), noms de color amb `get_color` |
| `my_labeling.py` | Part 3: cercador combinat (exemple: `"Pink dress"`) |
| `utils.py` | Model de color (`get_color_prob`, `rgb2gray`, …) |
| `utils_data.py` | Càrrega d’imatges, visualitzacions |
| `images/` | `train/`, `test/`, `gt.json` |

## Com executar-ho

Des de la carpeta del projecte:

```bash
# Proves ràpides (parseig + KNN + K-means + una cerca curta)
python3 my_labeling.py --test

# Cerca (anglès: color + peça). Recomanat limitar imatges si no vols esperar
python3 my_labeling.py "White shirts" --limite 100

# Ajuda
python3 my_labeling.py --help
```

Provar el KNN des de Python:

```python
from KNN import read_dataset, KNN

train_imgs, train_cls, _, test_imgs, test_cls, _ = read_dataset()
knn = KNN(train_imgs, train_cls, k=5)
pred = knn.predict(test_imgs[:10])
```

Provar K-means en una imatge:

```python
from KNN import read_dataset
from Kmeans import Kmeans, imagen_a_puntos_rgb

train_imgs, _, _, _, _, _ = read_dataset()
X = imagen_a_puntos_rgb(train_imgs[0])
km = Kmeans(X, K=3)
km.fit()
print(km.get_color())
```

## Consultes de text suportades

- **Colors (anglès):** Red, Orange, Brown, Yellow, Green, Blue, Purple, Pink, Black, Grey, White  
- **Peces (paraules clau):** dress, shirt, jeans, shorts, socks, sandals, flip flops, handbag, heels, …

Exemples: `"Pink dress"`, `"black sandals"`, `"Blue jeans"`.

## Llicència / ús

Codi per a l’assignatura; les imatges i `gt.json` pertanyen al material del curs.
