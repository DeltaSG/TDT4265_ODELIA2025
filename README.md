# README

## Oversikt

Dette repoet inneholder kode for klassifisering av 3D MR-bilder i tre klasser:

- Normal
- Benign
- Malignant

Modellen trener på flere MR-sekvenser per pasient og bruker PyTorch.

---

## Filer

### `data.py`

Håndterer data og preprocessing.

Inneholder:

- Leser inn treningsdata og testdata
- Lager datasett til PyTorch `DataLoader`
- Leser `.nii.gz` MR-volumer
- Normalisering og augmentering
- Splitter treningsdata i train/validation

---

### `model.py`

Definerer nevrale nettverk brukt i prosjektet.

Inneholder:

- `ResNet18` (3D versjon)
- `DenseNet121` (3D versjon)

Begge modeller tar inn 5 MR-kanaler og predikerer 3 klasser.

---

### `fit.py`

Brukes til trening av modellen.

Inneholder:

- Laster data
- Lager train/validation loaders
- Initialiserer modell
- Trener over flere epochs
- Evaluerer på validation-sett
- Lagrer beste modell i `checkpoints/`

---

### `eval.py`

Brukes til inferens / prediksjon på testsett.

Inneholder:

- Laster lagret modell
- Kjører prediksjoner på testdata
- Lager `predictions.csv`

---

### `EDA.py`

Exploratory Data Analysis.

Inneholder analyser og plots av datasettet:

- Antall bilder per sykehus
- Klassefordeling
- Antall kanaler
- Intensitetsfordeling
- Diverse figurer lagret i `plots/`

---

## Typisk bruk

Tren modell:

```bash
python fit.py
