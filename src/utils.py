from typing import Union, Tuple


def unpair(x: Union[int, Tuple[int,int]])->int:
    """If x is a tuple, return x[0] and x[1], else return x """
    if isinstance(x, tuple):
        return x[0]
    else:
        return x
    
def pair(x: Union[int, Tuple[int,int]])->Tuple[int,int]:
    """If x is a tuple, return x, else return (x,x) """
    if isinstance(x, tuple):
        return x
    else:
        return (x,x)