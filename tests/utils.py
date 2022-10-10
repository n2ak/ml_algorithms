def equal(a,b,t=1e-3,print_ok=False):
    import numpy as np
    mean = np.abs((a-b).mean())
    if print_ok:
        print("a",a)
        print("b",b)
        print("mean",mean)
    return mean <= t