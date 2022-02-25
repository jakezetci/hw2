from lsp import lstsq
import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt


if __name__ == "__main__":
    n = 500
    m = 20
    times = 10000
    A = np.random.rand(n, m)
    x = np.random.rand(m)
    Ax = A @ x
    scale = 0.01
    B = np.random.normal(loc=Ax, scale=scale, size=(times, n))
    error_arr = np.empty(times)
    fig, ax = plt.subplots(1, 1)

    for i, b in enumerate(B):
        x, cost, var = lstsq(A, b, 'svd')
        error_arr[i] = cost
    ax.hist(error_arr, bins=100, density=True)
    
    df = n - m
    mean, var, skew, kurt = chi2.stats(df, moments='mvsk')


    x = np.linspace(chi2.ppf(0.01, df),
                chi2.ppf(0.99, df), )
    '''ax.plot(x, chi2.pdf(x, df),
            'r-', lw=5, alpha=0.6, label='chi2 pdf')'''
