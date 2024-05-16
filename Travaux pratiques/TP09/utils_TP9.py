import numpy as np
from matplotlib import pyplot as plt


def plot_decision_multi(x1_min, x1_max, x2_min, x2_max, prediction, sample = 300):
    """Uses Matplotlib to plot and fill a region with 2 colors
    corresponding to 2 classes.

    Parameters
    ----------
    x1_min : float
        Minimum value for the first feature
    x1_max : float
        Maximum value for the first feature
    x2_min : float
        Minimum value for the second feature
    x2_max : float
        Maximum value for the second feature
    prediction :  (x : 2D vector) -> label : int
        Prediction function for a vector x
    sample : int, optional
        Number of samples on each feature (default is 300)
    """
    x1_list = np.linspace(x1_min, x1_max, sample)
    x2_list = np.linspace(x2_min, x2_max, sample)
    y_grid_pred = [[prediction(np.array([x1,x2])) for x1 in x1_list] for x2 in x2_list] 
    l = np.shape(np.unique(y_grid_pred))[0] - 1
    plt.contourf(x1_list, x2_list, y_grid_pred, levels=l, colors=plt.rcParams['axes.prop_cycle'].by_key()['color'], alpha=0.35)

