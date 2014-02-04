import numpy as np

def generate_filter(filter_name,filter_size,**kwargs):
    """ This function generate filter specified by the parameters
    Parameters
    ----------
    filter_name: string<'box','gaussian'>
        the name of filter
    filter_size: tuple, shape(1,2)
        the size of filter
    **kwargs: sigma for Gaussian filter
    """
    # Obtain filter size
    n_row,n_col = filter_size
    
    # Generate box filter
    if filter_name == 'box':
        res = np.ones((n_row,n_col))*1./n_row/n_col
    
    # Generate Gaussian filter
    elif filter_name == 'gaussian':
        len_x = (n_col-1)/2
        len_y = (n_row-1)/2
        assert len(kwargs) == 1
        for key,value in kwargs.iteritems():
            sigma = value
        x,y = np.meshgrid(range(-len_x,len_x+1),range(-len_y,len_y+1))
        res = np.exp(-(x**2+y**2)/(2*sigma**2))
        res = res/np.sum(res)
    
    # Generate 1d derivative of Gaussian
    elif filter_name == 'diff_gaussian':
        len_x = (n_col-1)/2
        assert n_row == 1
        assert len(kwargs) == 1
        for key,value in kwargs.iteritems():
            sigma = value
        x = np.array(range(-len_x,len_x+1))
        res = -x*np.exp(-x**2/(2*sigma**2))/(2*sigma**3*np.sqrt(2*np.pi))
        #res = res/np.sum(abs(res))
    else:
        print "ERROR: Filter Undefined"

    return res

if __name__ == "__main__":
    res = generate_filter('gaussian',(5,5),sigma=1.4)
    print res
