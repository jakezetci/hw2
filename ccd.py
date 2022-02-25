from astropy.io import fits
from lsp import lstsq
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json

      
if __name__ == "__main__":
    with fits.open('ccd.fits') as hdul:
        hdul.verify('fix')
        data = hdul[0].data
    data = np.asarray(data, dtype=np.int32)

    data_better = (data[:, 1, :, :] - data[:, 0, :, :])
    sigma_x = np.var(data_better, axis=(1, 2))
    x = np.mean(data, axis=(1, 2, 3))
    style_default = matplotlib.font_manager.FontProperties()
    style_default.set_size('xx-large')
    style_default.set_family(['Calibri', 'Helvetica', 'Arial', 'serif'])
    fig = plt.figure(figsize=(20, 12))
    ax = fig.add_subplot(1, 1, 1)
    np.append(sigma_x, 1)
    A = np.vstack((x, np.ones_like(x)))
    args, cost, var = lstsq(A.T, sigma_x, 'svd', rcond=1e-6)
    datapoints, = plt.plot(x, sigma_x, 'o', markersize=3)
    linear, = plt.plot(x, x*args[0]+args[1], lw=2)
    ax.set_xlabel('общая засветка', fontproperties=style_default)
    ax.set_ylabel('выборочная дисперсия', fontproperties=style_default)
    datapoints.set_label('посчитанные точки')
    linear.set_label('линейное приближение')
    plt.legend(loc='best', prop=style_default)
    plt.savefig('ccd.png')
    g = args[0]/2
    sigma_r = args[1]/g**2
    err_g = var[0, 0] / 2
    err_sigma = var[0, 0] - 2 * var[0][1] + 2 * var[1][1] #ковариацию не очень
    # знаю как считать
    d = {
        "ron": np.round(sigma_r, decimals=2),
        "ron_err": np.round(err_sigma, decimals=2),
        "gain": np.round(g, decimals=2),
        "gain_err": np.round(err_g, decimals=2)
        }
    with open('ccd.json', 'w') as f:
        json.dump(d, f, indent=2)
    