import numpy as np
import os

neg_xtalk_matrix = np.zeros((101, 101))
pos_xtalk_matrix = np.zeros((101, 101))

os.chdir("../")

for j1 in range(101):
    for j2 in range(101):
        path = f"matrix_elements/xtalk_{j1}_{j2}.txt"
        try:
            with open(path, "r") as f:
                line = f.readline().strip()
                result_list = line.split(",")
                neg_xtalk_matrix[j1, j2] = float(result_list[0])
                pos_xtalk_matrix[j1, j2] = float(result_list[1])
                print(f"Matrix elements [{j1},{j2}] retrieved.")

        except FileNotFoundError:
            neg_xtalk_matrix[j1, j2] = np.nan  # Mark missing elements
            pos_xtalk_matrix[j1, j2] = np.nan
            print(f"file for ({j1},{j2}) not found. Fill in with nan by default")

os.chdir("results")

# Save matrices to compressed binary file (fast and precise)
np.savez("xtalk_matrices_no_baseline.npz", neg_xtalk=neg_xtalk_matrix, pos_xtalk=pos_xtalk_matrix)
print("Matrices saved to xtalk_matrices.npz")

# Save readable CSV for both matrices
neg_csv_path = "xtalk_negative_no_baseline.csv"
pos_csv_path = "xtalk_positive_no_baseline.csv"
np.savetxt(neg_csv_path, neg_xtalk_matrix, delimiter=",", fmt="%.10f")
np.savetxt(pos_csv_path, pos_xtalk_matrix, delimiter=",", fmt="%.10f")
print(f"Matrices saved to {neg_csv_path} and {pos_csv_path}")

