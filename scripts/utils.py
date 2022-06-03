import numpy as np

def rand_mat(nrow,ncol):
    while True:
        M = np.zeros((nrow,ncol))
        for i in range(nrow):
            for j in range(ncol-1): M[i,j] = (1/ncol)+np.random.normal(0,0.2)
            M[i,ncol-1] = 1-sum(M[i,:])
        if (M>0).all(): break
    return M