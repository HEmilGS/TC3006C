import csv
import random
import math
from collections import Counter


def cargar_dataset(ruta, split=0.7):
    """Carga un dataset CSV y lo divide en train/test"""
    dataset = []
    with open(ruta, 'r') as f:
        reader = csv.reader(f)
        next(reader)  
        for row in reader:
            if len(row) == 0:
                continue
            dataset.append(row)

    for i in range(len(dataset)):
        dataset[i][:-1] = [float(x) for x in dataset[i][:-1]]

    random.shuffle(dataset)
    
    corte = int(len(dataset) * split)
    return dataset[:corte], dataset[corte:]


def distancia_euclidiana(p1, p2):
    """Calcula distancia euclidiana entre dos vectores (sin incluir la etiqueta)"""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1[:-1], p2[:-1])))

def predecir(train, nuevo, k=3):
    """Predice la clase de un nuevo punto con KNN"""
    distancias = [(distancia_euclidiana(nuevo, x), x[-1]) for x in train]
    distancias.sort(key=lambda x: x[0])
    vecinos = [label for _, label in distancias[:k]]
    return Counter(vecinos).most_common(1)[0][0]

def exactitud(test, predicciones):
    """Calcula el porcentaje de aciertos"""
    correctos = sum(1 for i in range(len(test)) if test[i][-1] == predicciones[i])
    return correctos / len(test) * 100

def main():

    train, test = cargar_dataset("winequality-red.csv")

    predicciones = []
    k = 5
    for row in test:
        salida = predecir(train, row, k)
        predicciones.append(salida)
        print(f"Real: {row[-1]}, Predicho: {salida}")

    acc = exactitud(test, predicciones)
    print(f"\nExactitud final: {acc:.2f}%")


if __name__ == "__main__":
    main()
   