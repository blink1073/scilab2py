from __future__ import absolute_import, print_function
import logging
import os
import signal

import numpy as np
import numpy.testing as test
from numpy.testing.decorators import skipif

from scilab2py import Scilab2Py, Scilab2PyError, get_log
from scilab2py.compat import StringIO

THIS_DIR = os.path.abspath(os.path.dirname(__file__))


class MiscTests(test.TestCase):

    def setUp(self):
        self.sci = Scilab2Py()
        self.sci.getd(THIS_DIR)

    def tearDown(self):
        self.sci.close()

    def test_unicode_docstring(self):
        '''Make sure unicode docstrings in Scilab functions work'''
        help(self.sci.test_datatypes)

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
        test.assert_raises(Scilab2PyError, self.sci._eval, "a='1")
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

    def test_prev_ans(self):
        self.sci._eval("5")
        assert self.sci._eval('_') == 5
