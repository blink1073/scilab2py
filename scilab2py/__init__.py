# -*- coding: utf-8 -*-
"""
Scilab2Py is a means to seamlessly call Scilab functions from Python.
It manages the Scilab session for you, sharing data behind the scenes using
MAT files.  Usage is as simple as:

.. code-block:: python

    >>> import scilab2py
    >>> sci = scilab2py.Scilab2Py()
    >>> x = sci.zeros(3,3)
    >>> print x, x.dtype
    [[ 0.  0.  0.]
     [ 0.  0.  0.]
     [ 0.  0.  0.]] float64

"""


__title__ = 'scilab2py'
__version__ = '0.1'
__author__ = 'Steven Silvester'
__license__ = 'MIT'
__copyright__ = 'Copyright 2014 Steven Silvester'
__all__ = ['Scilab2Py', 'Scilab2PyError', 'scilab', 'Struct', 'demo',
           'speed_test', 'thread_test', '__version__', 'get_log']


import imp
import functools
import os

from .core import Scilab2Py, Scilab2PyError
from .utils import Struct, get_log
from .demo import demo
from .speed_check import speed_test
from .thread_check import thread_test


try:
    scilab = Scilab2Py()
except Scilab2PyError as e:
    print(e)


def kill_scilab():
    """Kill all Scilab instances (cross-platform).

    This will restart the "Scilab" instance.  If you have instantiated
    Any other Scilab2Py objects, you must restart them.
    """
    import os
    if os.name == 'nt':
        os.system('taskkill /im Scilex /f')
    else:
        os.system('killall -9 scilab')
    scilab.restart()


# clean up namespace
del functools, imp, os
try:
    del core, utils, speed_check, thread_check
except NameError:  # pragma: no cover
    pass

