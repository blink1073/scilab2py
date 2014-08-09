from __future__ import absolute_import, print_function
import os
import pickle

import numpy as np
import numpy.testing as test

from scilab2py import Scilab2Py, Scilab2PyError
from scilab2py.utils import Struct

THIS_DIR = os.path.abspath(os.path.dirname(__file__))


class BasicUsageTest(test.TestCase):
    """Excercise the basic interface of the package
    """
    def setUp(self):
        self.sci = Scilab2Py()
        self.sci.getd(THIS_DIR)

    def test_run(self):
        """Test the run command
        """
        self.sci.run('y=ones(3,3)')
        y = self.sci.get('y')
        desired = np.ones((3, 3))
        test.assert_allclose(y, desired)
        self.sci.run('x = mean([[1, 2], [3, 4]])')
        x = self.sci.get('x')
        self.assertEqual(x, 2.5)
        self.assertRaises(Scilab2PyError, self.sci.run, '_spam')

    def test_call(self):
        """Test the call command
        """
        out = self.sci.call('ones', 1, 2)
        assert np.allclose(out, np.ones((1, 2)))
        U, S, V = self.sci.call('svd', [[1, 2], [1, 3]])
        assert np.allclose(U, ([[-0.57604844, -0.81741556],
                           [-0.81741556, 0.57604844]]))
        assert np.allclose(S,  ([[3.86432845, 0.],
                           [0., 0.25877718]]))
        assert np.allclose(V,  ([[-0.36059668, -0.93272184],
                           [-0.93272184, 0.36059668]]))
        out = self.sci.call('roundtrip.sci', 1)
        self.assertEqual(out, 1)
        fname = os.path.join(THIS_DIR, 'roundtrip.sci')
        out = self.sci.call(fname, 1)
        self.assertEqual(out, 1)
        self.assertRaises(Scilab2PyError, self.sci.call, '_spam')

    def test_put_get(self):
        """Test putting and getting values
        """
        self.sci.put('spam', [1, 2])
        out = self.sci.get('spam')
        assert np.allclose(out, np.array([1, 2]))
        self.sci.put(['spam', 'eggs'], ['foo', [1, 2, 3, 4]])
        spam, eggs = self.sci.get(['spam', 'eggs'])
        self.assertEqual(spam, 'foo')
        assert np.allclose(eggs, np.array([[1, 2, 3, 4]]))
        self.assertRaises(Scilab2PyError, self.sci.put, '_spam', 1)
        self.assertRaises(Scilab2PyError, self.sci.get, '_spam')

    def test_dynamic(self):
        """Test the creation of a dynamic function
        """
        tests = [self.sci.zeros, self.sci.ones, self.sci.plot]
        for item in tests:
            self.assertEqual(repr(type(item)), "<type 'function'>")
        self.assertRaises(Scilab2PyError, self.sci.__getattr__, 'aaldkfasd')
        self.assertRaises(Scilab2PyError, self.sci.__getattr__, '_foo')
        self.assertRaises(Scilab2PyError, self.sci.__getattr__, 'foo\W')

    def test_open_close(self):
        """Test opening and closing the Scilab session
        """
        sci_ = Scilab2Py()
        sci_.close()
        self.assertRaises(Scilab2PyError, sci_.put, names=['a'],
                          var=[1.0])
        sci_.restart()
        sci_.put('a', 5)
        a = sci_.get('a')
        assert a == 5

    def test_struct(self):
        """Test Struct construct
        """
        test = Struct()
        test.spam = 'eggs'
        test.eggs.spam = 'eggs'
        self.assertEqual(test['spam'], 'eggs')
        self.assertEqual(test['eggs']['spam'], 'eggs')
        test["foo"]["bar"] = 10
        self.assertEqual(test.foo.bar, 10)
        p = pickle.dumps(test)
        test2 = pickle.loads(p)
        self.assertEqual(test2['spam'], 'eggs')
        self.assertEqual(test2['eggs']['spam'], 'eggs')
        self.assertEqual(test2.foo.bar, 10)

    def test_syntax_error(self):
        """Make sure a syntax error in Scilab throws an Scilab2PyError
        """
        sci = Scilab2Py()
        self.assertRaises(Scilab2PyError, sci._eval, "a='1")
        sci = Scilab2Py()
        self.assertRaises(Scilab2PyError, sci._eval, "a=1+*3")

        sci.put('a', 1)
        a = sci.get('a')
        self.assertEqual(a, 1)

    def test_scilab_error(self):
        sci = Scilab2Py()
        self.assertRaises(Scilab2PyError, sci.run, 'a = ones2(1)')

    def test_context_manager(self):
        '''Make sure Scilab2Py works within a context manager'''
        with self.sci as sci1:
            ones = sci1.ones(1)
        assert ones == np.ones(1)
        with self.sci as sci2:
            ones = sci2.ones(1)
        assert ones == np.ones(1)

    def test_narg_out(self):
        s = self.sci.svd(np.array([[1, 2], [1, 3]]))
        assert s.shape == (2, 1)
        U, S, V = self.sci.svd([[1, 2], [1, 3]])
        assert U.shape == S.shape == V.shape == (2, 2)

    def test_temp_dir(self):
        sci = Scilab2Py(temp_dir='.')
        thisdir = os.path.dirname(os.path.abspath('.'))
        assert sci._reader.out_file.startswith(thisdir)
        assert sci._writer.in_file.startswith(thisdir)
