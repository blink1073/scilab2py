"""
scilab2py_test - Test value passing between python and Scliab.

Known limitations
-----------------
* The following Numpy array types cannot be sent directly via a MAT file.  The
float16/96/128 and complex192/256 can be recast as float64 and complex128.
   ** float16('e')
   ** float96('g')
   ** float128
   ** complex192('G')
   ** complex256
   ** read-write buffer('V')
"""
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


THIS_DIR = os.path.abspath(os.path.dirname(__file__))

if not os.name == 'nt':
    # needed for testing support
    if not hasattr(sys.stdout, 'buffer'):  # pragma: no cover
        class Dummy(object):

            def write(self):
                pass
        try:
            sys.stdout.buffer = Dummy()
        except AttributeError:
            pass


class ConversionTest(test.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sci = Scilab2Py()

    @classmethod
    def tearDownClass(cls):
        cls.sci.close()

    def test_python_conversions(self):
        """Test roundtrip python type conversions
        """
        for out_type, sci_type, in_type in TYPE_CONVERSIONS:
            if out_type == dict:
                outgoing = dict(x=1)
            elif out_type is None:
                outgoing = None
            else:
                outgoing = out_type(1)
            incoming, scilab_type = self.sci.roundtrip(outgoing)
            if type(incoming) != in_type:
                assert in_type(incoming) == incoming
            assert type(incoming) == in_type


class IncomingTest(test.TestCase):
    """Test the importing of all Scilab data types, checking their type

    Uses test_datatypes.sci to read in a dictionary with all Scilab types
    Tests the types of all the values to make sure they were
        brought in properly.

    """
    @classmethod
    def setUpClass(cls):
        with Scilab2Py() as sci:
            sci.getd(THIS_DIR)
            cls.data = sci.test_datatypes()

    def helper(self, base, keys, types):
        """
        Perform type checking of the values

        Parameters
        ==========
        base : dict
            Sub-dictionary we are accessing.
        keys : array-like
            List of keys to test in base.
        types : array-like
            List of expected return types for the keys.

        """
        for key, type_ in zip(keys, types):
            if not type(base[key]) == type_:
                try:
                    assert type_(base[key]) == base[key]
                except ValueError:
                    assert np.allclose(type_(base[key]), base[key])

    def test_int(self):
        """Test incoming integer types
        """
        keys = ['int8', 'int16', 'int32',
                'uint8', 'uint16', 'uint32']
        types = [np.int8, np.int16, np.int32,
                 np.uint8, np.uint16, np.uint32]
        self.helper(self.data.num.int, keys, types)

    def test_floats(self):
        """Test incoming float types
        """
        keys = ['double', 'complex', 'complex_matrix']
        types = [np.float64, np.complex128, np.ndarray]
        self.helper(self.data.num, keys, types)
        self.assertEqual(self.data.num.complex_matrix.dtype,
                         np.dtype('complex128'))

    def test_misc_num(self):
        """Test incoming misc numeric types
        """
        keys = ['matrix', 'vector', 'column_vector', 'matrix3d']
        types = [np.ndarray, np.ndarray, np.ndarray,
                      np.ndarray]
        self.helper(self.data.num, keys, types)

    def test_string(self):
        """Test incoming string types
        """
        keys = ['basic']
        types = [unicode]
        self.helper(self.data.string, keys, types)

    def test_mixed_struct(self):
        '''Test mixed struct type
        '''
        keys = ['array', 'cell', 'scalar']
        types = [list, str, float]
        self.helper(self.data.mixed, keys, types)


class RoundtripTest(test.TestCase):
    """Test roundtrip value and type preservation between Python and Scilab.

    Uses test_datatypes.sci to read in a dictionary with all Scilab types
    uses roundtrip.sci to send each of the values out and back,
        making sure the value and the type are preserved.

    """
    @classmethod
    def setUpClass(cls):
        cls.sci = Scilab2Py()
        cls.sci.getd(THIS_DIR)
        cls.data = cls.sci.test_datatypes()

    @classmethod
    def tearDownClass(cls):
        cls.sci.close()

    def nested_equal(self, val1, val2):
        """Test for equality in a nested list or ndarray
        """
        if isinstance(val1, list):
            for (subval1, subval2) in zip(val1, val2):
                if isinstance(subval1, list):
                    self.nested_equal(subval1, subval2)
                elif isinstance(subval1, np.ndarray):
                    np.allclose(subval1, subval2)
                else:
                    self.assertEqual(subval1, subval2)
        elif isinstance(val1, np.ndarray):
            np.allclose(val1, np.array(val2))
        elif isinstance(val1, (str, unicode)):
            self.assertEqual(val1, val2)
        else:
            try:
                assert (np.alltrue(np.isnan(val1)) and
                        np.alltrue(np.isnan(val2)))
            except (AssertionError, NotImplementedError):
                self.assertEqual(val1, val2)

    def helper(self, outgoing, expected_type=None):
        """
        Use roundtrip.sci to make sure the data goes out and back intact.

        Parameters
        ==========
        outgoing : object
            Object to send to Scilab.

        """
        incoming = self.sci.roundtrip(outgoing)
        if expected_type is None:
            expected_type = type(outgoing)
        self.nested_equal(incoming, outgoing)
        try:
            self.assertEqual(type(incoming), expected_type)
        except AssertionError:
            if type(incoming) == np.float32 and expected_type == np.float64:
                pass

    def test_int(self):
        """Test roundtrip value and type preservation for integer types
        """
        for key in ['int8', 'int16', 'int32',
                    'uint8', 'uint16', 'uint32']:
            self.helper(self.data.num.int[key], np.float64)

    def test_float(self):
        """Test roundtrip value and type preservation for float types
        """
        for key in ['double', 'complex', 'complex_matrix']:
            self.helper(self.data.num[key])

    def test_misc_num(self):
        """Test roundtrip value and type preservation for misc numeric types
        """
        for key in ['matrix', 'vector', 'column_vector',  'matrix3d']:
            self.helper(self.data.num[key])

    def test_string(self):
        """Test roundtrip value and type preservation for string types
        """
        for key in ['basic']:
            self.helper(self.data.string[key])

    def test_cell_array(self):
        """Test roundtrip value and type preservation for cell array types
        """
        for key in ['array']:
            self.helper(self.data.cell[key])
        #self.helper(DATA.cell['array'], np.ndarray)

    def test_scilab_origin(self):
        '''Test all of the types, originating in scilab, and returning
        '''
        self.sci.run('x = test_datatypes()')
        self.sci.put('y', self.data)

        for key in self.data.keys():
            if isinstance(self.data[key], dict):
                for subkey in self.data[key].keys():
                    if subkey == 'int':
                        continue
                    cmd = 'isequal(x.{0}.{1},y.{0}.{1})'.format(key, subkey)
                    assert self.sci.run(cmd) == 'T'
                continue
            else:
                cmd = 'isequal(x.{0},y.{0})'.format(key)
                assert self.sci.run(cmd) == 'T'


class BuiltinsTest(test.TestCase):
    """Test the exporting of standard Python data types, checking their type.

    Runs roundtrip.sci and tests the types of all the values to make sure they
    were brought in properly.

    """
    @classmethod
    def setUpClass(cls):
        cls.sci = Scilab2Py()
        cls.sci.getd(THIS_DIR)

    @classmethod
    def tearDownClass(cls):
        cls.sci.close()

    def helper(self, outgoing, incoming=None, expected_type=None):
        """
        Uses roundtrip.sci to make sure the data goes out and back intact.

        Parameters
        ==========
        outgoing : object
            Object to send to Scilab
        incoming : object, optional
            Object already retreived from Scilab

        """
        if incoming is None:
            incoming = self.sci.roundtrip(outgoing)
        if not expected_type:
            for out_type, _, in_type in TYPE_CONVERSIONS:
                if out_type == type(outgoing):
                    expected_type = in_type
                    break
        if not expected_type:
            expected_type = np.ndarray
        try:
            self.assertEqual(incoming, outgoing)
        except ValueError:
            assert np.allclose(np.array(incoming), np.array(outgoing))
        if type(incoming) != expected_type:
            incoming = self.sci.roundtrip(outgoing)
            try:
                assert np.allclose(expected_type(incoming), incoming)
            except TypeError:
                assert expected_type(incoming), incoming

    def test_set(self):
        """Test python set type
        """
        test = set((1, 2, 3, 3))
        incoming = self.sci.roundtrip(test)
        assert np.allclose(tuple(test), incoming)
        self.assertEqual(type(incoming), np.ndarray)

    def test_tuple(self):
        """Test python tuple type
        """
        test = tuple((1, 2, 3))
        self.helper(test, expected_type=np.ndarray)

    def test_list(self):
        """Test python list type
        """
        tests = [[1, 2], [3, 4]]
        self.helper(tests[0])
        self.helper(tests[1], expected_type=list)

    def test_list_of_tuples(self):
        """Test python list of tuples
        """
        test = [(1, 2), (1.5, 3.2)]
        self.helper(test)

    def test_numeric(self):
        """Test python numeric types
        """
        test = np.random.randint(1000)
        self.helper(int(test))
        self.helper(long(test))
        self.helper(float(test))
        self.helper(complex(1, 2))

    def test_string(self):
        """Test python str and unicode types
        """
        tests = ['spam', unicode('eggs')]
        for test in tests:
            self.helper(test)

    def test_nested_list(self):
        """Test python nested lists
        """
        test = [[1, 2], [3, 4]]
        self.helper(test)
        incoming = self.sci.roundtrip(test)
        for i in range(len(test)):
            assert np.alltrue(incoming[i] == np.array(test[i]))

    def test_bool(self):
        """Test boolean values
        """
        tests = (True, False)
        for t in tests:
            incoming = self.sci.roundtrip(t)
            self.assertEqual(incoming, t)
            self.assertEqual(incoming.dtype, np.float)

    def test_none(self):
        """Test sending None type
        """
        incoming = self.sci.roundtrip(None)
        assert np.isnan(incoming)


class NumpyTest(test.TestCase):
    """Check value and type preservation of Numpy arrays
    """
    codes = np.typecodes['All']
    blacklist_codes = ['V', 'M', 'f', 'F']
    blacklist_names = ['float128', 'float96', 'complex192', 'complex256']

    @classmethod
    def setUpClass(cls):
        cls.sci = Scilab2Py()
        cls.sci.getd(THIS_DIR)

    @classmethod
    def tearDownClass(cls):
        cls.sci.close()

    def test_scalars(self):
        """Send scalar numpy types and make sure we get the same number back.
        """
        for typecode in self.codes:
            if typecode in self.blacklist_codes:
                continue
            outgoing = (np.random.randint(-255, 255) + np.random.rand(1))
            try:
                outgoing = outgoing.astype(typecode)
            except TypeError:
                continue
            if outgoing.dtype.name in self.blacklist_names:
                continue
            incoming = self.sci.roundtrip(outgoing)
            if outgoing.dtype.str in ['<M8[us]', '<m8[us]']:
                outgoing = outgoing.astype(np.uint64)
            if outgoing.dtype.kind in 'bui':
                outgoing = outgoing.astype(np.float)
            try:
                assert np.allclose(incoming, outgoing)
            except (ValueError, TypeError, NotImplementedError,
                    AssertionError):
                assert np.alltrue(np.array(incoming).astype(typecode) ==
                                  outgoing)

    def test_ndarrays(self):
        """Send ndarrays and make sure we get the same array back
        """
        for typecode in self.codes:
            ndims = 2
            size = [np.random.randint(1, 10) for i in range(ndims)]
            outgoing = (np.random.randint(-255, 255, tuple(size)))
            try:
                outgoing += np.random.rand(*size).astype(outgoing.dtype,
                                                         casting='unsafe')
            except TypeError:  # pragma: no cover
                outgoing += np.random.rand(*size).astype(outgoing.dtype)
            if typecode in ['U', 'S', 'O']:
                continue
            else:
                try:
                    outgoing = outgoing.astype(typecode)
                except TypeError:
                    continue
            if (typecode in self.blacklist_codes or
                    outgoing.dtype.name in self.blacklist_names):
                continue
            incoming = self.sci.roundtrip(outgoing)
            incoming = np.array(incoming)
            if outgoing.size == 1:
                outgoing = outgoing.squeeze()
            if len(outgoing.shape) > 2 and 1 in outgoing.shape:
                incoming = incoming.squeeze()
                outgoing = outgoing.squeeze()
            elif incoming.size == 1:
                incoming = incoming.squeeze()
            assert incoming.shape == outgoing.shape
            if outgoing.dtype.str in ['<M8[us]', '<m8[us]']:
                outgoing = outgoing.astype(np.uint64)
            try:
                assert np.allclose(incoming, outgoing)
            except (AssertionError, ValueError, TypeError,
                    NotImplementedError):
                if 'c' in incoming.dtype.str:
                    incoming = np.abs(incoming)
                    outgoing = np.abs(outgoing)
                assert np.alltrue(np.array(incoming).astype(typecode) ==
                                  outgoing)

    def test_empty(self):
        '''Test roundtrip empty matrices
        '''
        empty = np.empty((100, 100))
        incoming, type_ = self.sci.roundtrip(empty)
        assert empty.squeeze().shape == incoming.squeeze().shape
        assert np.allclose(empty[np.isfinite(empty)],
                           incoming[np.isfinite(incoming)])
        assert type_ == 'constant'

    def test_mat(self):
        '''Verify support for matrix type
        '''
        test = np.random.rand(1000)
        test = np.mat(test)
        incoming, type_ = self.sci.roundtrip(test)
        assert np.allclose(test, incoming)
        assert test.dtype == incoming.dtype
        assert type_ == 'constant'

    def test_masked(self):
        '''Test support for masked arrays
        '''
        test = np.random.rand(100)
        test = np.ma.array(test)
        incoming, type_ = self.sci.roundtrip(test)
        assert np.allclose(test, incoming)
        assert test.dtype == incoming.dtype
        assert type_ == 'constant'

    def test_infinite(self):
        """Test support for inf and nan types.
        """
        x = np.NaN
        y = np.inf
        self.sci.put('x', x)
        self.sci.put('y', y)
        assert np.isnan(self.sci.get('x'))
        assert np.isinf(self.sci.get('y'))
        self.sci.run('a = %nan')
        self.sci.run('b= %inf')
        assert np.isnan(self.sci.get('a'))
        assert np.isinf(self.sci.get('b'))


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


class MiscTests(test.TestCase):

    def setUp(self):
        self.sci = Scilab2Py()
        self.sci.getd(THIS_DIR)

    def tearDown(self):
        self.sci.close()

    def test_unicode_docstring(self):
        '''Make sure unicode docstrings in Scilab functions work'''
        help(self.sci.test_datatypes)

    def test_context_manager(self):
        '''Make sure Scilab2Py works within a context manager'''
        with self.sci as sci1:
            ones = sci1.ones(1)
        assert ones == np.ones(1)
        with self.sci as sci2:
            ones = sci2.ones(1)
        assert ones == np.ones(1)

    def test_logging(self):
        # create a stringio and a handler to log to it
        def get_handler():
            sobj = StringIO()
            hdlr = logging.StreamHandler(sobj)
            hdlr.setLevel(logging.DEBUG)
            return hdlr
        hdlr = get_handler()
        self.sci.logger.addHandler(hdlr)

        # generate some messages (logged and not logged)
        self.sci.ones(1, verbose=True)

        self.sci.logger.setLevel(logging.DEBUG)
        self.sci.zeros(1)
        # check the output
        lines = hdlr.stream.getvalue().strip().split('\n')
        resp = '\n'.join(lines)
        assert 'zeros(A__)' in resp
        assert '0.0' in resp
        assert lines[0].startswith('loadmatfile')

        # now make an object with a desired logger
        logger = get_log('test')
        hdlr = get_handler()
        logger.addHandler(hdlr)

        logger.setLevel(logging.INFO)
        sci2 = Scilab2Py(logger=logger)
        # generate some messages (logged and not logged)
        sci2.ones(1, verbose=True)
        sci2.logger.setLevel(logging.DEBUG)
        sci2.zeros(1)
        sci2.close()

        # check the output
        lines = hdlr.stream.getvalue().strip().split('\n')
        resp = '\n'.join(lines)
        assert 'zeros(A__)' in resp
        assert '0.' in resp
        assert lines[0].startswith('loadmatfile')

    def test_demo(self):
        from scilab2py import demo
        try:
            demo.demo(0.01, interactive=False)
        except AttributeError:
            demo(0.01, interactive=False)

    def test_remove_files(self):
        from scilab2py.utils import _remove_temp_files
        _remove_temp_files()

    def test_threads(self):
        from scilab2py import thread_test
        thread_test()

    def test_plot(self):
        self.sci.figure()
        self.sci.plot([1, 2, 3])
        self.sci.close_()

    def test_narg_out(self):
        s = self.sci.svd(np.array([[1, 2], [1, 3]]))
        assert s.shape == (2, 1)
        U, S, V = self.sci.svd([[1, 2], [1, 3]])
        assert U.shape == S.shape == V.shape == (2, 2)

    def test_help(self):
        help(self.sci)

    def test_trailing_underscore(self):
        x = self.sci.ones_()
        assert np.allclose(x, np.ones(1))

    def test_using_closed_session(self):
        with Scilab2Py() as sci:
            sci.close()
            test.assert_raises(Scilab2PyError, sci.call, 'ones')

    def test_pause(self):
        self.assertRaises(Scilab2PyError,
                                      lambda:  self.sci.xpause(10e6, timeout=3))

    def test_func_without_docstring(self):
        out = self.sci.test_nodocstring(5)
        assert out == 5
        return
        # TODO: fill this in when we make help commands for funcs
        assert 'user-defined function' in self.sci.test_nodocstring.__doc__
        assert THIS_DIR in self.sci.test_nodocstring.__doc__

    def test_func_noexist(self):
        test.assert_raises(Scilab2PyError, self.sci.call, 'Scilab2Py_dummy')

    def test_timeout(self):
        with Scilab2Py(timeout=2) as sci:
            sci.xpause(2.1e6, timeout=5)
            test.assert_raises(Scilab2PyError, sci.xpause, 4e6)

    def test_call_path(self):
        with Scilab2Py() as sci:
            sci.getd(THIS_DIR)
            DATA = sci.call('test_datatypes.sci')
        assert DATA.string.basic == 'spam'

        with Scilab2Py() as sci:
            path = os.path.join(THIS_DIR, 'test_datatypes.sci')
            DATA = sci.call(path)
        assert DATA.string.basic == 'spam'

    def test_long_variable_name(self):
        name = 'this_variable_name_is_over_32_char'
        self.sci.put(name, 1)
        x = self.sci.get(name)
        assert x == 1

    def test_syntax_error_embedded(self):
        test.assert_raises(Scilab2PyError, self.sci.run, """eval("a='1")""")
        self.sci.put('b', 1)
        x = self.sci.get('b')
        assert x == 1

    def test_oned_as(self):
        x = np.ones(10)
        self.sci.put('x', x)
        assert self.sci.get('x').shape == x[:, np.newaxis].T.shape
        sci = Scilab2Py(oned_as='column')
        sci.put('x', x)
        assert sci.get('x').shape == x[:, np.newaxis].shape

    def test_temp_dir(self):
        sci = Scilab2Py(temp_dir='.')
        thisdir = os.path.dirname(os.path.abspath('.'))
        assert sci._reader.out_file.startswith(thisdir)
        assert sci._writer.in_file.startswith(thisdir)

    @skipif(not hasattr(signal, 'alarm'))
    def test_interrupt(self):

        def receive_signal(signum, stack):
            raise KeyboardInterrupt

        signal.signal(signal.SIGALRM, receive_signal)

        signal.alarm(5)
        self.sci.run("xpause(10e6);kladjflsd")

        self.sci.put('c', 10)
        x = self.sci.get('c')
        assert x == 10

    def test_clear(self):
        """Make sure clearing variables does not mess anything up."""
        self.sci.clear()


if __name__ == '__main__':  # pragma: no cover
    print('Scilab2Py test')
    print('*' * 20)
    test.run_module_suite()
