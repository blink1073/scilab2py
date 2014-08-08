"""Send a numpy array roundtrip to Scilab using a script.
"""
from scilab2py import scilab
import numpy as np

if __name__ == '__main__':
    x = np.array([[1, 2], [3, 4]], dtype=float)
    out, oclass = scilab.roundtrip(x)
    # or scilab.call('roundtrip', x)
    # or scilab.call('roundtrip.m', x)
    # or scilab.call('/path/to/roundtrip.m', x)
    import pprint
    pprint.pprint([x, x.dtype, out, oclass, out.dtype])
