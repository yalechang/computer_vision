import numpy as np

def ransac_homography(src,dst,n_iter_max=100):
    """
    Parameters
    ----------
    src: array, shape(N,2)
    dst: array, shape(N,2)

    Returns
    -------
    h: array,shape(3,3)
    """
    
    mtr_a = np.zeros((2*src.shape[0],9))
    for i in range(src.shape[0]):
        [x1,y1,x2,y2] = [src[i,0],src[i,1],dst[i,0],dst[i,1]]
        mtr_a[i*2,:] = np.array([x1,y1,1,0,0,0,-x1*x2,-y1*x2,-x2])
        mtr_a[i*2+1,:] = np.array([0,0,0,x1,y1,1,-x1*y2,-y1*y2,-y2])

