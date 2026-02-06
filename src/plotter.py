import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_confusion_matrix(y_true, y_pred, k):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    plt.title(f"Confusion Matrix (K = {k})")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks([0, 1], ["No Purchase", "Purchase"])
    plt.yticks([0, 1], ["No Purchase", "Purchase"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center",  color="white" if cm[i, j] > 2000 else "black")

    plt.tight_layout()
    plt.savefig(f"confusion_matrix_k{k}.png")
    plt.close()