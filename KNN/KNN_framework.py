import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def main():
    df = pd.read_csv("winequality-red.csv")

    # Convertir calidad a binaria
    df["quality"] = df["quality"].apply(lambda x: "Bueno" if x >= 6 else "Malo")

    X = df.drop("quality", axis=1)
    y = df["quality"]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 80%
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y, test_size=0.4, random_state=42, shuffle=True
    )
    # 20%
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, shuffle=True
    )

    resultados = []

    k_values = [1, 5, 9]

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, weights="distance") #distance/uniform
        knn.fit(X_train, y_train)

        y_train_pred = knn.predict(X_train)
        y_val_pred = knn.predict(X_val)
        y_test_pred = knn.predict(X_test)

        acc_train = accuracy_score(y_train, y_train_pred) * 100
        acc_val = accuracy_score(y_val, y_val_pred) * 100
        acc_test = accuracy_score(y_test, y_test_pred) * 100

        resultados.append({
            "k": k,
            "Train (%)": acc_train,
            "Validation (%)": acc_val,
            "Test (%)": acc_test
        })

        print(f"\n===== Resultados con k={k} =====")
        print(f"Train Accuracy: {acc_train:.2f}%")
        print(f"Validation Accuracy: {acc_val:.2f}%")
        print(f"Test Accuracy: {acc_test:.2f}%")
        print("\nReporte de clasificación en test:")
        print(classification_report(y_test, y_test_pred))

    tabla = pd.DataFrame(resultados)
    print("\n===== Resumen completo =====")
    print(tabla.to_string(index=False))

    n_groups = len(k_values)
    train_acc = tabla["Train (%)"].values
    val_acc = tabla["Validation (%)"].values
    test_acc = tabla["Test (%)"].values

    bar_width = 0.25
    index = np.arange(n_groups)

    plt.figure(figsize=(8,5))
    plt.bar(index, train_acc, bar_width, color='skyblue', label='Train')
    plt.bar(index + bar_width, val_acc, bar_width, color='lightgreen', label='Validation')
    plt.bar(index + 2*bar_width, test_acc, bar_width, color='salmon', label='Test')

    plt.xlabel('k')
    plt.ylabel('Exactitud (%)')
    plt.title('Exactitud de KNN por conjunto y valor de k')
    plt.xticks(index + bar_width, k_values)
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(axis='y')
    plt.show()

if __name__ == "__main__":
    main()
