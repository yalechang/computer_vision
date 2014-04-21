import numpy as np

def get_subwindow(im,pos,sz):
    """This function extract subwindow from an array and add paddings for edge
    handling.

    Parameters
    ----------
    im: array, shape(row,column)
        image with size row, column

    pos: list
        location of center in the form of [idx_row,idx_col]

    sz: int
        radius of patches
    """
    row,col = pos
    row_range = np.array(range(row-sz,row+sz+1))
    col_range = np.array(range(col-sz,col+sz+1))
    row_range[row_range<0] = 0
    row_range[row_range>im.shape[0]-1] = im.shape[0]-1
    col_range[col_range<0] = 0
    col_range[col_range>im.shape[1]-1] = im.shape[1]-1
    
    """
    out = []
    for i in row_range:
        for j in col_range:
            out.append(im[i,j])
    """
    yy,xx = np.meshgrid(row_range,col_range)
    return im[yy,xx].T

    #return np.array(out).reshape(sz*2+1,sz*2+1)

if __name__ == "__main__":
    im = np.arange(36).reshape(6,6)
    pos = [1,2]
    sz = 1
    x = get_subwindow(im,pos,sz)

