import numpy as np
    

def rand_mat(nrow,ncol):
    while True:
        M = np.zeros((nrow,ncol))
        for i in range(nrow):
            for j in range(ncol-1): M[i,j] = (1/ncol)+np.random.normal(0,0.2)
            M[i,ncol-1] = 1-sum(M[i,:])
        if (M>0).all(): break
    return M


def delta(M1, M2):
    if M1.shape != M2.shape: raise ValueError("shape mismatch")
    if len(M1.shape) == 1:
        M3 = np.zeros(M1.shape[0])
        for i in range(M3.shape[0]): M3[i] = (M1[i] - M2[i])**2
    else:              
        M3 = np.zeros(M1.shape)
        for i in range(M1.shape[0]): M3[i] = np.array([(M1[i,j] - M2[i,j])**2 for j in range(M1.shape[1])])
    return M3.sum()