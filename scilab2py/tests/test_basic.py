from __future__ import absolute_import, print_function
import logging
import os
import pickle
import signal
import sys

import numpy as np
import numpy.testing as test
from numpy.testing.decorators import skipif

from scilab2py import Scilab2Py, Scilab2PyError, get_log
from scilab2py.utils import Struct
from scilab2py.compat import unicode, long, StringIO



TYPE_CONVERSIONS = [
    (int, 'constant', np.float64),
    (long, 'constant', np.float64),
    (float, 'constant', np.float64),
    (complex, 'constant', np.complex128),
    (str, 'string', np.unicode_),
    (unicode, 'string', np.unicode_),
    (bool, 'constant', np.float64),
    (None, 'constant', np.float64),
    (np.int8, 'constant', np.float64),
    (np.int16, 'constant', np.float64),
    (np.int32, 'constant', np.float64),
    (np.int64, 'constant', np.float64),
    (np.uint8, 'constant', np.float64),
    (np.uint16, 'constant', np.float64),
    (np.uint32, 'constant', np.float64),
    (np.float32, 'constant', np.float64),
    (np.float64, 'constant', np.float64),
    (np.str, 'string', np.unicode_),
    (np.double, 'constant', np.float64),
    (np.complex64, 'constant', np.complex128),
    (np.complex128, 'constant', np.complex128),
]

'''
Test incoming cell array types ... FAIL
Test incoming float types ... ERROR
Test incoming integer types ... ERROR
Test incoming logical type ... ERROR
Test incoming misc numeric types ... FAIL
Test mixed struct type ... FAIL
Test incoming string types ... FAIL
Test incoming struct array types ... ERROR
Test demo
test_pause (scilab2py.tests.test_scilab2py.MiscTests) ... ERROR
test_plot (scilab2py.tests.test_scilab2py.MiscTests) ... ERROR
test_remove_files (scilab2py.tests.test_scilab2py.MiscTests) ... ERROR
Make sure a singleton sparse matrix works ... ERROR
test_threads (scilab2py.tests.test_scilab2py.MiscTests) ... ERROR
test_timeout (scilab2py.tests.test_scilab2py.MiscTests) ... FAIL

'''

class ConversionTest(test.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sci = Scilab2Py()

    @classmethod
    def tearDownClass(cls):
        cls.sci.close()


    def test_narg_out(self):
        s = self.sci.svd(np.array([[1, 2], [1, 3]]))
        assert s.shape == (2, 1)
        U, S, V = self.sci.svd([[1, 2], [1, 3]])
        assert U.shape == S.shape == V.shape == (2, 2)

