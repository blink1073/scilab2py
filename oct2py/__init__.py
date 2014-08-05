# -*- coding: utf-8 -*-
"""
Scilab2Py is a means to seamlessly call M-files and GNU Octave functions from Python.
It manages the Octave session for you, sharing data behind the scenes using
MAT files.  Usage is as simple as:

.. code-block:: python

    >>> import oct2py
    >>> oc = oct2py.Scilab2Py() 
    >>> x = oc.zeros(3,3)
    >>> print x, x.dtype
    [[ 0.  0.  0.]
     [ 0.  0.  0.]
     [ 0.  0.  0.]] float64

If you want to run legacy m-files, do not have MATLAB(TM), and do not fully
trust a code translator, this is your library.  
"""


__title__ = 'oct2py'
__version__ = '1.6.0'
__author__ = 'Steven Silvester'
__license__ = 'MIT'
__copyright__ = 'Copyright 2014 Steven Silvester'
__all__ = ['Scilab2Py', 'Scilab2PyError', 'octave', 'Struct', 'demo', 'speed_test',
           'thread_test', '__version__', 'get_log']


import imp
import functools
import os

from .session import Scilab2Py, Scilab2PyError
from .utils import Struct, get_log
from .demo import demo
from .speed_check import speed_test
from .thread_check import thread_test


try:
    scilab = Scilab2Py()
except Scilab2PyError as e:
    print(e)


def kill_octave():
    """Kill all octave instances (cross-platform).

    This will restart the "octave" instance.  If you have instantiated
    Any other Scilab2Py objects, you must restart them.
    """
    import os
    if os.name == 'nt':
        # TODO: what is Windows name?
        os.system('taskkill /im octave /f')
    else:
        os.system('killall -9 scilab')
    scilab.restart()


# clean up namespace
del functools, imp, os
try:
    del session, utils, speed_check, thread_check
except NameError:  # pragma: no cover
    pass

