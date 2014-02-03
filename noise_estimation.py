import numpy as np

def noise_estimation(img):
    """This function estimates noise from a sequence of images
    Parameters
    ----------
    img: array, shape(n_image,n_row,n_col)
        array containing a sequence of images, which can be used to estimate
        noise

    Returns
    ------
    img_noise: array, shape(n_row,n_col)
        noise level at each pixel
    """
    n_image,n_row,n_col = img.shape

    # Average of images
    img_avg = np.average(img,0)

    # Variance of pixel values
    tmp = np.zeros((n_row,n_col))
    for i in range(n_image):
        tmp += (img[i,:,:]-img_avg)**2
    noise_est = np.sqrt(tmp/(n_image-1))
    
    return noise_est
