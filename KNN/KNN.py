import csv
import random
import math
from collections import Counter

def cargar_dataset(ruta, split=0.7, binario=False):
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

        if binario:
            calidad = int(dataset[i][-1])
            dataset[i][-1] = "Bueno" if calidad >= 6 else "Malo"


    random.shuffle(dataset)

    corte = int(len(dataset) * split)
    return dataset[:corte], dataset[corte:]

def normalizar(dataset):
    columnas = list(zip(*[row[:-1] for row in dataset]))
    mins = [min(col) for col in columnas]
    maxs = [max(col) for col in columnas]

    for row in dataset:
        for i in range(len(row) - 1): 
            if maxs[i] - mins[i] != 0:
                row[i] = (row[i] - mins[i]) / (maxs[i] - mins[i])
    return dataset

def distancia_euclidiana(p1, p2):
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
    train, test = cargar_dataset("winequality-red.csv", binario=True)

    train = normalizar(train)
    test = normalizar(test)

    resultados_k = {}

    for k in [1, 5, 9]:
        print(f"\n===== Resultados con k={k} =====")
        predicciones = []
        for row in test:
            salida = predecir(train, row, k)
            predicciones.append(salida)
            print(f"Real: {row[-1]}  |  Predicho: {salida}")

        acc = exactitud(test, predicciones)
        resultados_k[k] = acc

    print("\n===== Resumen de exactitudes por k =====")
    for k, acc in resultados_k.items():
        print(f"k={k} -> Exactitud: {acc:.2f}%")

if __name__ == "__main__":
    main()
