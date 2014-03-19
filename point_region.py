import numpy as np

def line_two_points(points):
    """ This script returns the coefficients of a line given two points
    Parameters
    ----------
    points: array, shape(2,2)
        coordinates of two points, where the first row represents point_a and
        the second row represents point_b

    Returns
    -------
    coef: array, shape(1,3)
        (a,b,c) in linear equation given by ax + by + c = 0
    """
    
    assert points.shape == (2,2)
    # If the line is parallel to y axis
    if points[0,0] == points[1,0]:
        return np.array([1,0,-points[0,0]])
    # If the line has finite slope
    else:
        a = points[0,1]-points[1,1]
        b = points[1,0]-points[0,0]
        c = points[0,0]*points[1,1]-points[1,0]*points[0,1]
        return np.array([1.,b*1./a,c*1./a])

def point_region(bound,point):
    """ Tell if a point lies in the region specified by four bounding points
    Parameters
    ----------
    bound: array, shape(4,2)
        coordinates of four bounding points

    point: array, shape(1,2)
        coordinate of the point

    Returns
    -------
    flag: bool
        flag == True means point lies in the region specified by bound
        otherwise not
    """
    line_1 = line_two_points(bound[[0,1],:])
    line_2 = line_two_points(bound[[1,2],:])
    line_3 = line_two_points(bound[[2,3],:])
    line_4 = line_two_points(bound[[3,0],:])
    
    x = np.hstack((point,1))
    # Test if point lies between line_1 and line_3
    flag_1 = (line_1.dot(x))*(line_3.dot(x))

    # Test if point lies between line_2 and line_4
    flag_2 = (line_2.dot(x))*(line_4.dot(x))

    if flag_1<0 and flag_2<0:
        flag = True
    else:
        flag = False
    #print x,line_1,line_2,line_3,line_4
    #print flag_1,flag_2
    return flag

if __name__ == "__main__":
    bound = np.array([[0,1],[1,0],[2,1],[1,2]])
    point = np.array([1,0.5])
    print point_region(bound,point)
