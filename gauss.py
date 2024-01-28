import numpy as np
import time
import csv

def gauss_jordan_inverse(matrice):
    n = len(matrice)
    matrice_augmentee = np.hstack([matrice, np.identity(n)])

    for i in range(n):

        ligne_pivot = max(range(i, n), key=lambda k: abs(matrice_augmentee[k, i]))
        matrice_augmentee[[i, ligne_pivot]] = matrice_augmentee[[ligne_pivot, i]]

        matrice_augmentee[i] /= matrice_augmentee[i, i]
        for j in range(n):
            if i != j:
                matrice_augmentee[j] -= matrice_augmentee[j, i] * matrice_augmentee[i]
    matrice_inverse = matrice_augmentee[:, n:]

    return matrice_inverse

tailles_matrices = [2, 3, 10, 15, 20, 100, 200, 300, 500]

nom_fichier_csv = "resultats_gauss.csv"

with open(nom_fichier_csv, mode='w', newline='') as fichier_csv:

    writer = csv.writer(fichier_csv)


    writer.writerow(["Taille de la matrice", "Matrice", "Matrice Inverse", "Temps d'exécution (secondes)"])

    for n in tailles_matrices:
        matrice = np.random.rand(n, n)

        debut = time.time()

        matrice_inverse = gauss_jordan_inverse(matrice)

        fin = time.time()

        np.set_printoptions(threshold=np.inf, linewidth=np.inf)

        writer.writerow([f"Matrice {n}x{n}", matrice, matrice_inverse, f"Temps d'exécution: {fin - debut:.6f} secondes"])

        np.set_printoptions(threshold=1000, linewidth=75)

        writer.writerow(["\n\n\n"])
