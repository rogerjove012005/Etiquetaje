# Informe — Projecte Etiquetador

**Autor:** Roger Jové (`rogerjove012005`)  
**Repositori:** [Etiquetaje](https://github.com/rogerjove012005/Etiquetaje)  
**Objectiu:** etiquetar imatges de productes per **forma** (KNN) i **color** (K-means + model de color) i permetre cerques combinades en llenguatge natural simplificat.

---

## Entrega informe

Aquest document resumeix el lliurament del projecte en tres parts alineades amb l’enunciat:

| Part | Fitxer principal | Contingut |
|------|------------------|-----------|
| 1 | `KNN.py` | Càrrega de dades, classificador k-NN sobre píxels en escala de grisos |
| 2 | `Kmeans.py` | K-means en RGB, WCD, `find_bestK` (criteri del 20%), `get_color` |
| 3 | `my_labeling.py` | Cercador combinat (consulta text → filtre per forma KNN + color K-means) |

Eines i dades: Python 3, NumPy, PIL, Matplotlib (visualitzacions a `utils_data.py`). Imatges i `gt.json` a `images/`.

---

## Desenvolupament

### 2.1 Arquitectura general

1. **Representació per forma:** imatge RGB → escala de grisos (`rgb2gray` a `utils.py`) → vector de 4.800 dimensions (80×60) per comparar amb distància euclidiana.
2. **Classificació de forma:** KNN amb vot majoritari entre els `k` veïns més propers (`KNN.py`).
3. **Color:** cada píxel és un punt (R,G,B); K-means agrupa i els centroides són colors predominants; `get_color_prob` (llibreria del curs a `utils.py`) assigna nom entre 11 colors bàsics.
4. **Elecció de K (nombre de grups de color):** es calcula la distància intra-classe (WCD, suma de quadrats punt–centroide) per a cada K; es tria el primer `k` tal que la millora relativa en passar de `k` a `k+1` sigui **inferior al 20%**, tal com indica l’enunciat.
5. **Cercador:** es parseja una frase en anglès (paraula de prenda + nom de color); per cada imatge es comprova si la predicció KNN coincideix amb la prenda i si el color demanat apareix entre els noms retornats pels centroides després del K-means.

### 2.2 Decisions de disseny

- **`read_dataset`:** es reutilitza `utils_data.read_dataset` des de `KNN.py` amb ruta per defecte `./images/gt.json` (el projecte no usa `test/gt.json`).
- **KNN:** implementació explícita (bucles de vot) per llegibilitat; distàncies vectoritzades per fila de test contra tot el train.
- **K-means:** assignació vectoritzada (matriu de distàncies) per eficiència; actualització de centroides per mitjana per clúster; clúster buit → recentre aleatori.
- **`my_labeling.py`:** **submostreig de píxeles** (`salto`) abans del K-means per reduir temps en el cercador sense canviar la idea del mètode.

### 2.3 Anàlisi de paràmetres (mínim tres, segons enunciat)

1. **Valor de `k` al KNN:** amb `k` petit el model és més sensible al soroll; amb `k` gran es suavitza la frontera però es pot barrejar classes si la regió és heterogènia. En aquest projecte s’usa `k=5` per defecte (compromís habitual).
2. **Inicialització dels centroides (K-means):** centroides inicials = `K` píxels aleatoris del conjunt (`semilla=42` per repetibilitat). Diferents llavors poden canviar lleugerament WCD i noms de color si dos centroides són similars.
3. **Espai de color i noms:** el clustering és en **RGB**; el nom del color ve del model **CIELab + funcions de pertinença** a `get_color_prob`, no d’una distància simple en RGB. Això acosta els noms als 11 colors del catàleg però no garanteix coincidència amb les etiquetes humanes del JSON.
4. **Cerca de la millor K (`find_bestK`):** el llindar del 20% fixa quan deixar d’afegir clústers; si `max_K` és massa petit, pot quedar-se en un K subòptim; si és massa gran, el cost creix linealment en el nombre de valors de K provats (cada prova entrena un K-means complet).

---

## Anàlisi d’eficiència

### 3.1 Mesures (entorn de referència)

Conjunt: **2.328** imatges train, **851** test; imatge 80×60×3 (4.800 píxels). Mesures orientatives amb `time.perf_counter()`:

| Operació | Temps aproximat |
|----------|-------------------|
| Inicialitzar KNN (vectors train) | ~0,12 s |
| Predir 50 imatges test | ~0,36 s |
| Predir **totes** les 851 imatges test | ~5,7 s |
| Un `fit` K-means (K=4, 4.800 punts) | ~0,008 s |
| `find_bestK` (fins K=8) sobre una imatge | ~0,09 s |
| `buscar` 100 imatges test (`salto=3`) | ~2,0 s |
| `buscar` **tot** el test sense límit (`salto=2`, una consulta tipus *Blue jeans*) | ~25 s |

### 3.2 Complexitat asimptòtica (ordre de magnitud)

- **KNN (predir una imatge):** per cada imatge de test es calcula la distància a les **N** imatges train en **d** dimensions → cost per imatge **O(N·d)**. Aquí `d=4800`, `N=2328` → el coll d’ampolla principal en volum de dades.
- **K-means (una imatge, T iteracions, K clústers, n píxels):** cada iteració assignació **O(n·K·3)** amb la implementació vectoritzada (nombre de punts × clústers × 3 components); actualització **O(n)**. `find_bestK` prova diversos valors de K → multiplica el cost per un factor lineal en el rang de K.
- **Cercador `buscar`:** per cada imatge revisada: una predicció KNN + un o més K-means (incloent `find_bestK` si s’usa). Cost global **O(M · (cost KNN + cost K-means))** on M és el nombre d’imatges recorregudes (tot el test o el `limite`).

### 3.3 Millores aplicades

- Vectorització de distàncies a centroides a `Kmeans.fit`.
- **Submostreig** (`salto`) a `my_labeling.py` per reduir `n` al K-means en el cercador.
- Opció **`--limite N`** des de terminal per proves ràpides sense recórrer les 851 imatges.

### 3.4 Conclusions

El sistema compleix el flux de l’enunciat (forma supervisada, color no supervisat + noms, cerca combinada). El **KNN sobre tot el train** domina el temps quan es processa molt test; el **K-means per imatge** és ràpid en solitari, però es repeteix moltes vegades al cercador, per això el submostreig i el límit són importants per a un ús interactiu. Futura millora: precomputar etiquetes de color/forma per a tot el catàleg i fer la cerca sobre una taula indexada en lloc de tornar a entrenar K-means a cada consulta.

---

## Referències internes del codi

- `KNN.py` — `read_dataset`, classe `KNN`, `predict`
- `Kmeans.py` — `imagen_a_puntos_rgb`, classe `Kmeans`, `fit`, `withinClassDistance`, `find_bestK`, `get_color`
- `my_labeling.py` — `parsear_consulta`, `buscar`, proves `python3 my_labeling.py --test`
- `utils.py` / `utils_data.py` — funcions del material del curs
