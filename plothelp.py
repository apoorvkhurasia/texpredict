from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn
from matplotlib.figure import Figure


def plot_confusion_matrix(data: np.ndarray, labels: List[str],
                          figsize: Tuple[int, int], dpi: int,
                          title: str) -> Figure:
    """
    Plot confusion matrix using heatmap.
    Args:
        :param data:    ndarray of shape (n_classes, n_classes)
                        Confusion matrix whose i-th row and j-th
                        column entry indicates the number of
                        samples with true label being i-th class
                        and prediced label being j-th class.
        :param labels:  Labels which will be plotted across axes.
        :param figsize: The figure size to use.
        :param dpi:     The DPI of the figure.
        :param title:   The title of the figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    seaborn.set(color_codes=True)
    seaborn.set(font_scale=0.7)
    ax.set_title(title)
    ax = seaborn.heatmap(data, annot=True, cmap="YlGnBu",
                         fmt=".1g", cbar_kws={'label': 'Scale'})

    latex_labels = list(map(lambda s: "$" + s.split(sep=";")[0] + "$", labels))
    ax.set_xticklabels(latex_labels)
    ax.set_yticklabels(latex_labels)
    ax.set(ylabel="True Label", xlabel="Predicted Label")
    return fig
