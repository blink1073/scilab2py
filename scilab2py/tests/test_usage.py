from __future__ import absolute_import, print_function
import logging
import os
import pickle
import signal

import numpy as np
import numpy.testing as test
from numpy.testing.decorators import skipif

from scilab2py import Scilab2Py, Scilab2PyError, get_log
from scilab2py.utils import Struct
from scilab2py.compat import StringIO

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
            sci.xpause(2.1e6, timeout=20)
            test.assert_raises(Scilab2PyError, sci.xpause, 10e6)

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

        signal.alarm(10)
        self.sci.run("xpause(20e6);kladjflsd")

        self.sci.put('c', 10)
        x = self.sci.get('c')
        assert x == 10

    def test_clear(self):
        """Make sure clearing variables does not mess anything up."""
        self.sci.clear()
