import numpy as np

def line_supercover(y0, x0, y1, x1):
    """
    Adapted from:
    https://github.com/scikit-image/scikit-image/issues/2232
    https://gist.github.com/amccaugh/f459e45650915351bb65070141a28e3f
    """

    dx = np.abs(x1-x0)
    dy = np.abs(y1-y0)
    x = x0
    y = y0
    n = dx + dy
    err = dx - dy

    if x1 > x0:
        x_inc = 1
    else:
        x_inc = -1
    if y1 > y0:
        y_inc = 1
    else:
        y_inc = -1

    max_length = (max(dx,dy)+1)*3
    rr = np.zeros((max_length), dtype=np.int)
    cc = np.zeros((max_length), dtype=np.int)

    dx = 2 * dx
    dy = 2 * dy

    ii = 0
    while n > 0:
        rr[ii] = y
        cc[ii] = x
        ii = ii + 1
        if (err > 0):
            x += x_inc
            err -= dy
        elif (err < 0):
            y += y_inc
            err += dx
        else: # If err == 0 the algorithm is on a corner
            rr[ii] = y + y_inc
            cc[ii] = x
            rr[ii+1] = y
            cc[ii+1] = x + x_inc
            ii = ii + 2
            x += x_inc
            y += y_inc
            err = err + dx - dy
            n = n - 1
        n = n - 1
    rr[ii] = y
    cc[ii] = x

    return np.asarray(rr[0:ii+1]), np.asarray(cc[0:ii+1])

def demo():
    import matplotlib.pyplot as plt
    x, y = line_supercover(0,0,2,9)
    img  = np.zeros((10, 10))
    img[x,y] = 1.
    plt.imshow(img)
