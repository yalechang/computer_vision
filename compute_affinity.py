import numpy as np
import numpy.linalg as la
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import pairwise_distances

def compute_affinity(data,flag_sigma='local',sigma=100.,nn=7):
 
    # Find local sigma for each sample
    if flag_sigma == 'local':
        
        # Compute pairwise Euclidean distances between samples
        dis_data = pairwise_distances(data)

        # Sort distances
        dis_data_sorted = np.sort(dis_data)

        # Find sigma, set it to be nn nearest neighbor
        sigma = dis_data_sorted[:,nn]
        
        # Compute kernel matrix
        aff_data = np.zeros((data.shape[0],data.shape[0]))
        for i in range(data.shape[0]):
            for j in range(i,data.shape[0]):
                tmp = np.sum((data[i,:]-data[j,:])**2)
                aff_data[i,j] = np.exp(-tmp/sigma[i]*sigma[j])
                aff_data[j,i] = aff_data[i,j]
    
    # Find global sigma for all the sample
    elif flag_sigma == 'global':
        
        # Compute pairwise Euclidean distances between samples
        dis_data = pairwise_distances(data)

        # Take the median value as scale parameter
        sigma = np.median(dis_data)**2
        aff_data = rbf_kernel(data,data,gamma=1./sigma)
        tmp = np.diag(1./np.sqrt(np.sum(aff_data,1)))
        aff_data = tmp.dot(aff_data).dot(tmp)

    elif flag_sigma == 'manual':
        aff_data = rbf_kernel(data,data,gamma=1./sigma)
    
    else:
        print "ERROR of flag_sigma"

    return (sigma,aff_data)
