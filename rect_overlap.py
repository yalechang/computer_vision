import numpy as np

def rect_overlap(rect_1,rect_2):
    """ Compute Area of intersection between two rectanges

    Parameters
    ----------
    rect_1: list,len(3)
        left-top vertex, width, height

    rect_2: list,len(3)
        left_top vertex, width, height

    Returns
    -------
    area: int
        area of intersection between two rectanges
    """

    p_1,width_1,height_1 = rect_1
    p_2,width_2,height_2 = rect_2
    
    r1_left = p_1[0]
    r1_right = p_1[0]+width_1
    r1_top = p_1[1]
    r1_bottom = p_1[1]+height_1

    r2_left = p_2[0]
    r2_right = p_2[0]+width_2
    r2_top = p_2[1]
    r2_bottom = p_2[1]+height_2

    left = max(r1_left,r2_left)
    right = min(r1_right,r2_right)
    bottom = min(r1_bottom,r2_bottom)
    top = max(r1_top,r2_top)

    if left<right and bottom>top:
        area = (right-left)*(bottom-top)
    else:
        area = 0
    return area

if __name__ == "__main__":
    rect_1 = [[10,10],10,10]
    rect_2 = [[15,5],10,10]
    print rect_overlap(rect_1,rect_2)

