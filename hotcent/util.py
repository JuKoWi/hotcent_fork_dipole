import numpy as np

def tail_smoothening(x, y):
    """ For given grid-function y(x), make smooth tail.
    
    Aim is to get (e.g. for Slater-Koster tables and repulsions) smoothly
    behaving energies and forces near cutoff region.
    
    Make is such that y and y' go smoothly exactly to zero at last point.
    Method: take largest neighboring points y_k and y_(k+1)  (k<N-3) such
    that line through them passes zero below x_(N-1). Then fit
    third-order polynomial through points y_k, y_k+1 and y_N-1.
    
    Return:
    smoothed y-function on same grid.
    """
    if np.all(abs(y) < 1e-10):
        return y

    N = len(x)
    xmax = x[-1]

    for i in range(N - 3, 1, -1):
        x0i = x[i] - y[i]
        x0i /= (y[i + 1] - y[i]) / (x[i + 1] - x[i])
        if x0i < xmax:
            k = i
            break

    if k < N / 4:
        for i in range(N):
            print(x[i], y[i])
        msg = 'Problem with tail smoothening: requires too large tail.'
        raise RuntimeError(msg)

    if k == N - 3:
        y[-1] = 0.
        return y

    else:
        # g(x) = c2 * (xmax - x )** m + c3 * (xmax - x) ** (m + 1) goes through 
        # (xk, yk), (xk + 1, yk + 1) and (xmax, 0)
        # Try different m if g(x) should change sign (this we do not want)
        sgn = np.sign(y[k])
        for m in range(2, 10):
            a1, a2 = (xmax - x[k]) ** m, (xmax - x[k]) ** (m + 1)
            b1, b2 = (xmax - x[k + 1]) ** m, (xmax - x[k + 1]) ** (m + 1)
            c3 = (y[k] - a1 * y[k + 1] / b1) / (a2 - a1 * b2 / b1)
            c2 = (y[k] - a2 * c3) / a1

            for i in range(k + 2,N):
                y[i] = c2 * (xmax - x[i]) ** 2 + c3 * (xmax - x[i]) ** 3

            y[-1] = 0.  # once more excplicitly
            if np.all(y[k:] * sgn >= 0):
                break

            if m == 9:
                msg = 'Problems with function smoothening; need new algorithm?'
                raise RuntimeError(msg)
    return y
