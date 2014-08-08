""" Get a sample of all datatypes from Scilab and print the result
"""
from scilab2py import scilab

if __name__ == '__main__':
    out = scilab.test_datatypes()
    import pprint
    pprint.pprint(out)
