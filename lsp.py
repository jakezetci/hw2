import numpy as np


def lstsq_ne(a, b):
    a = np.atleast_2d(a)
    b = np.atleast_1d(b)
    x = np.linalg.solve(a.T@a, a.T@b)
    cost = a @ x - b
    var = cost/(b.shape[0]-x.shape[0])*np.eye(a.shape[0])
    return x, cost, var

def lstsq_svd(a, b, rcond=None):
    a = np.atleast_2d(a)
    b = np.atleast_1d(b)
    u, s, vh = np.linalg.svd(a, full_matrices=False)
    if rcond is None:
        where = (s != 0.0)
    else:
        where = s > s[0] * rcond
    x = vh.T @ np.divide(u.T[:s.shape[0],:] @ b, s, out=np.zeros(a.shape[1]), where=where)
    r = a @ x - b
    cost = np.inner(r, r)
    sigma0 = cost / (b.shape[0] - x.shape[0])
    var = vh.T @ np.diag(s**(-2)) @ vh * sigma0    
    return x, cost, var

def lstsq(a, b, method, **kwargs):
    if method == 'ne':
        return lstsq_ne(a, b)
    elif method == 'svd':
        return lstsq_svd(a, b, **kwargs)
    else:
        return None
