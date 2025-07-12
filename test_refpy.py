'''
Test Function
'''

import doctest
import refpy

def run_doctests():
    '''
    Run doctests for the refpy classes
    '''
    result1 = doctest.testmod(
        refpy.pipe_properties, verbose=False,
        optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS
    )
    result2 = doctest.testmod(
        refpy.dnv_limit_states, verbose=False,
        optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS
    )
    result3 = doctest.testmod(
        refpy.lateral_buckling, verbose=False,
        optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS
    )
    result4 = doctest.testmod(
        refpy.pipe_soil_interaction, verbose=False,
        optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS
    )
    total_attempted = (
        result1.attempted +
        result2.attempted +
        result3.attempted +
        result4.attempted
    )
    total_failed = (
        result1.failed +
        result2.failed +
        result3.failed +
        result4.failed
    )
    total_passed = total_attempted - total_failed
    print(
        f"Doctest summary: {total_attempted} attempted,"
        f"{total_passed} passed,"
        f"{total_failed} failed."
    )


if __name__ == "__main__":
    run_doctests()
