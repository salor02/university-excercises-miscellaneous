#!/usr/bin/python3
import sys

def max_difference_between_files(file1_path, file2_path):
    # Leggi e converti i numeri dal primo file
    with open(file1_path, 'r') as f1:
        values1 = []
        for line in f1:
            values1.extend(map(float, line.split()))

    # Leggi e converti i numeri dal secondo file
    with open(file2_path, 'r') as f2:
        values2 = []
        for line in f2:
            values2.extend(map(float, line.split()))

    # Trova la lunghezza minima per confrontare solo le coppie corrispondenti
    min_len = min(len(values1), len(values2))

    # Calcola la differenza massima tra le coppie di valori
    max_diff = max(abs(values1[i] - values2[i]) for i in range(min_len))

    print(f"Max abs diff bewteen {file1_path} and {file2_path}: {max_diff}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python confronto.py file1.txt file2.txt")
        sys.exit(1)
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    max_difference_between_files(file1, file2)
