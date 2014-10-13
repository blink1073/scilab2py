
******************
Information
******************

Dynamic Functions
=================
ScilabPy will create methods for you on the fly, which correspond to Scilab
functions.  For example:

.. code-block:: python

    >>> from scilab2py import scilab
    >>> scilab.getd('/path/to/')
    >>> scilab.myscript()

Additionally, you can look up the documentation for one of these methods using
`help`, which will launch the Scilab Help Browser.

.. code-block:: python

    >>> from scilab2py import scilab
    >>> scilab.help('ones')


Interactivity
=============
Scilab2Py supports code completion in IPython, so once you have created a method,
you can recall it on the fly, so scilab.one<TAB> would give you ones.
Structs (mentioned below) also support code completion for attributes.

You can share data with an Scilab session explicitly using the `push` and
`pull` methods.  When using other Scilab2Py methods, the variable names in Scilab
start with underscores because they are temporary.  Note that integer values in python
are converted to floats prior to sending to Scilab.

.. code-block:: python

    >>> from scilab2py import scilab
    >>> scilab.push('a', 1)
    >>> scilab.pull('a')
    1.0


Direct Interaction
==================
Scilab2Py supports the Scilab `pause` function
which drops you into an interactive Scilab prompt in the current session.


Logging
=======
Scilab2Py supports logging of session interaction.  You can provide a logger
to the constructor or set one at any time.

.. code-block:: python

    >>> import logging
    >>> from scilab2py import Scilab2Py, get_log
    >>> sci = Scilab2Py(logger=get_log())
    >>> sci.logger = get_log('new_log')
    >>> sci.logger.setLevel(logging.INFO)

All Scilab2Py methods support a `verbose` keyword.  If True, the commands are
logged at the INFO level, otherwise they are logged at the DEBUG level.


Shadowed Function Names
=======================
If you'd like to call an Scilab function that is also an Scilab2Py method,
you must add a trailing underscore. Only the `exit` and `eval` functions are affected.
For example:

.. code-block:: python

    >>> from scilab2py import scilab
    >>> scilab.eval_('2>1')
    1.0


Timeout
=======
Scilab2Py sessions have a `timeout` attribute that determines how long to wait
for a command to complete.  The default is 1e6 seconds (indefinite).
You may either set the timeout for the session, or as a keyword
argument to an individual command.  The session is closed in the event of a
timeout.


.. code-block:: python

    >>> from scilab2py import scilab
    >>> scilab.timeout = 3
    >>> scilab.sleep(2)
    >>> scilab.sleep(2, timeout=1)
    Traceback (most recent call last):
    ...
    scilab2py.utils.Scilab2PyError: Session timed out


Interruption
===============
Scilab2Py will catch a Keyboard Interrupt and interrupt the current Scilab command unless you
are on Windows, where it will restart the Scilab session.


Context Manager
===============
Scilab2Py can be used as a Context Manager.  The session will be closed and the
temporary m-files will be deleted when the Context Manager exits.

.. code-block:: python

    >>> from scilab2py import Scilab2Py
    >>> with Scilab2Py() as sci:
    >>>     sci.ones(10)


Nargout
=======
Scilab2Py handles nargout the same way that Scilab would (which is not how it
normally works in Python).  The number return variables affects the
behavior of the Scilab function.  For example, the following two calls to SVD
return different results:

.. code-block:: python

    >>> from scilab2py import scilab
    >>> out = scilab.svd(np.array([[1,2], [1,3]])))
    >>> U, S, V = scilab.svd([[1,2], [1,3]])


Structs
=======
Struct is a convenience class that mimics an Scilab structure variable type.
It is a dictionary with attribute lookup, and it creates sub-structures on the
fly of arbitrary nesting depth.  It can be pickled. You can also use tab
completion for attributes when in IPython.

.. code-block:: python

    >>> from scilab2py import Struct
    >>> test = Struct()
    >>> test['foo'] = 1
    >>> test.bizz['buzz'] = 'bar'
    >>> test
    {'foo': 1, 'bizz': {'buzz': 'bar'}}
    >>> import pickle
    >>> p = pickle.dumps(test)


Unicode
=======
Scilab2Py supports Unicode characters, so you may feel free to use scripts that
contain them.


Speed
=====
There is a performance penalty for passing information using MAT files.
If you have a lot of calculations, it is probably better to make an m-file
that does the looping and data aggregation, and pass that back to Python
for further processing.  To see an example of the speed penalty on your
machine, run:

.. code-block:: python

    >>> import scilab2py
    >>> scilab2py.speed_test()


Threading
=========
If you want to use threading, you *must* create a new `Scilab2Py` instance for
each thread.  The `scilab` convenience instance is in itself *not* threadsafe.
Each `Scilab2Py` instance has its own dedicated Scilab session and will not
interfere with any other session.


IPython Notebook
================
Scilab2Py provides ScilabMagic_ for IPython, including inline plotting in
notebooks.  This requires IPython >= 1.0.0.

.. _ScilabMagic: http://nbviewer.ipython.org/github/blink1073/scilab2py/blob/master/example/scilabmagic_extension.ipynb?create=1



