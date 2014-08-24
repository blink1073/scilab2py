Scilab2Py: Python to Scilab Bridge
===================================

.. image:: https://badge.fury.io/py/scilab2py.png/
    :target: http://badge.fury.io/py/scilab2py

.. image:: https://pypip.in/d/scilab2py/badge.png
        :target: https://crate.io/packages/scilab2py/

.. image:: https://coveralls.io/repos/blink1073/scilab2py/badge.png?branch=master
        :target: https://coveralls.io/r/blink1073/scilab2py?branch=master


Scilab2Py is a means to seamlessly call Scilab functions and scripts from Python.
It manages the Scilab session for you, sharing data behind the scenes using
MAT files.  Usage is as simple as:

.. code-block:: python

    >>> sci = scilab2py.Scilab2Py()
    >>> x = sci.zeros(3,3)
    >>> print x, x.dtype
    [[ 0.  0.  0.]
     [ 0.  0.  0.]
     [ 0.  0.  0.]] float64
    ...


Features
--------

- Supports most Scilab datatypes and most Python datatypes and Numpy dtypes.
- Provides ScilabMagic_ for IPython, including inline plotting in notebooks.
- Supports cell arrays and structs with arbitrary nesting.
- Supports sparse matrices.
- Builds methods on the fly linked to Scilab commands (e.g. `zeros` above).
- Nargout is automatically inferred by the number of return variables.
- Thread-safety: each Scilab2Py object uses an independent Scilab session.
- Can be used as a context manager.
- Supports Unicode characters.
- Supports logging of session commands.
- Optional timeout command parameter to prevent runaway Scilab sessions.


.. _ScilabMagic: http://nbviewer.ipython.org/github/blink1073/scilab2py/blob/master/example/scilabmagic_extension.ipynb?create=1


Installation
------------
You must have Scilab_ 5.4 or newer installed and in your PATH.
You must have the Numpy and Scipy libraries installed.

To install Scilab2Py, simply:

.. code-block:: bash

    $ pip install scilab2py


Documentation
-------------

Documentation is available online_.

For version information, see the Revision History_.

.. _online: http://blink1073.github.io/scilab2py
.. _Scilab: http://www.scilab.org/download/
.. _History: https://github.com/blink1073/scilab2py/blob/master/HISTORY.rst
