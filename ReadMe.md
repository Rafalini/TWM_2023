# Zliczanie i detekcja zachowań samochodów

W ramach projektu została wykonana aplikacja do detekcji i śledzenia samochodów. W implementacji jest wykorzystywana sieć YOLO v4, wraz z pre-trenowanymi wagami (niestety trening własnego modelu na google collabie nie pozwolił nam osiągnąć wyników osiąganych przez sieci trenowane na ImageNet'cie). Posługując się danymi z detekcji i śledzenia (tutaj korzystamy z IOU) wykonujemy analizę zachowań samochodów i wykrywany przykładowo zmianę pasa ruchu.

![image](https://github.com/Rafalini/TWM_2023/assets/44322872/4908111d-c9ab-4891-8360-e969eae542c9)

## Instrukcja uruchomienia programu
1. Pobrać dwa pliki z folderu dnn_model: yolov4.weights oraz yolov4.cfg z następującego dysku:
https://drive.google.com/drive/folders/1FsKN1mzJe3RKrgKS8E1nCbMmlMf0ouMH?usp=share_link
2. Pobrane pliki umieścić w folderze dnn_model repozytorium
3. Uruchomić plik object_tracking.py - zostanie uruchomiony skrypt dla przykładowego nagrania

## Inne wskazówki
Na dysku https://drive.google.com/drive/folders/1FsKN1mzJe3RKrgKS8E1nCbMmlMf0ouMH?usp=share_link znajduje się wykorzystywany zbiór danych (folder datasets) oraz przykładowe nagrania działania programu (folder examples).

