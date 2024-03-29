import matplotlib
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    n_NEs = [50, 100, 391, 491]

    FalseDef_avg_score = [38.01, 43.70, 51.40, 51.57]
    TrueDef_avg_score = [49.17, 50.45, 53.27, 53.30]

    plt.figure(figsize=(10, 5))
    plt.xticks(n_NEs)

    plt.grid(axis='y', linestyle='--', color='lightgray')
    plt.plot(n_NEs, FalseDef_avg_score, marker='o')
    plt.plot(n_NEs, TrueDef_avg_score, marker='o')
    plt.xlabel('Number distinct NEs')
    plt.ylabel('avg micro-F1')
    plt.title("Zero-Shot AVG micro-F1 on MIT/CrossNER", fontsize=14)
    plt.legend(['FalseDef', 'TrueDef'])
    plt.show()
    
    """

    n_samples_per_NE = [1, 5, 10]

    FalseDef_avg_score = [51.10, 51.57, 53.49]
    TrueDef_avg_score = [46.07, 53.30, 53.45]

    plt.figure(figsize=(10, 5))
    plt.xticks(n_samples_per_NE)

    plt.grid(axis='y', linestyle='--', color='lightgray')
    plt.plot(n_samples_per_NE, FalseDef_avg_score, marker='o')
    plt.plot(n_samples_per_NE, TrueDef_avg_score, marker='o')
    plt.yticks(range(45, 55, 1))

    plt.xlabel('Number samples per NEs')
    plt.ylabel('avg micro-F1')
    plt.title("Zero-Shot AVG micro-F1 on MIT/CrossNER", fontsize=14)
    plt.legend(['FalseDef', 'TrueDef'])
    plt.show()
    """