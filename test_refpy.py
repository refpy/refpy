'''
Test Function
'''

import doctest
import refpy

def run_doctests():
    '''
    Run doctests for the refpy module
    '''
    doctest.testmod(refpy.pipe_properties)
    doctest.testmod(refpy.lateral_buckling)
    doctest.testmod(refpy.pipe_soil_interaction,
                    optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)

if __name__ == "__main__":
    run_doctests()
