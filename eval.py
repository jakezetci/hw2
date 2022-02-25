from lsp import lstsq
import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt
import matplotlib


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
    cost_total = 0
    for i, b in enumerate(B):
        x, cost, var = lstsq(A, b, 'svd')
        error_arr[i] = cost
        cost_total += cost
    ax.hist(error_arr, bins=100, density=True)
    df = n - m
    mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

    maxerr = max(error_arr)
    minerr = min(error_arr)
    x = np.linspace(minerr, maxerr, num=100)
    xx = np.linspace(df-50, df+50, num=100)
    pdf, = ax.plot(x, chi2.pdf(xx, df)*1e4, #простите за подгон
            'r-', lw=5, alpha=0.6, label='chi2 pdf')
    style_default = matplotlib.font_manager.FontProperties()
    style_default.set_size('large')
    style_default.set_family(['Calibri', 'Helvetica', 'Arial', 'serif'])
    plt.savefig('chi2.png')
    pdf.set_label('теоретическое распределение')
    plt.legend(loc='best', prop=style_default)
