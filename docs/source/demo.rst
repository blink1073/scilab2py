
***********
Demo
***********

Output of Scliab2Py demo script, showing most of the features of the library.  Note that the two
plot commands will generate an interactive plot in the actual demo.
To run interactively:



mport numpy as np
    from scilab2py import Scilab2Py
    sci = Scilab2Py()
    # basic commands
    print(sci.abs(-1))
    print(sci.ones(3, 3))
    # plotting
    sci.plot([1,2,3],'-o')
    raw_input('Press Enter to continue...')
    sci.close_()
    xx = np.arange(-2*np.pi, 2*np.pi, 0.2)
    sci.surf(np.subtract.outer(np.sin(xx), np.cos(xx)))
    raw_input('Press Enter to continue...')
    sci.close_()
    # single vs. multiple return values
    print(sci.svd(np.array([[1,2], [1,3]])))
    U, S, V = sci.svd([[1,2], [1,3]])
    print(U, S, V)
    # low level constructs
    sci.run("y=ones(3,3)")
    print(sci.get("y"))
    sci.run("x=zeros(3,3)", verbose=True)
    x = sci.call('rand', 1, 4)
    print(x)
    t = sci.call('rand', 1, 2, verbose=True)
    y = np.zeros((3,3))
    sci.put('y', y)
    print(sci.get('y'))
    from scilab2py import Struct
    y = Struct()
    y.b = 'spam'
    y.c.d = 'eggs'
    print(y.c['d'])
    print(y)


.. code-block:: python

   >>> import scilab2py
   >>> scilab2py.demo()


    >>> import numpy as np
    >>> from scilab2py import Scilab2Py
    >>> sci = Scilab2Py()
    >>> # basic commands
    >>> print(sci.abs(-1))
    1.0
    >>> print(sci.ones(3, 3))
    [[ 1.  1.  1.]
     [ 1.  1.  1.]
     [ 1.  1.  1.]]
    >>> # plotting
    >>> sci.plot([1,2,3],'-o')
    Press Enter to continue...

.. image:: static/plot.png

.. code-block:: python

    >>> sci.close_()
    >>> xx = np.arange(-2*np.pi, 2*np.pi, 0.2)
    >>> sci.surf(np.subtract.outer(np.sin(xx), np.cos(xx)))
    Press Enter to continue...

.. image:: static/surf.png

.. code-block:: python

    >>> # single vs. multiple return values
    >>> print(sci.svd(np.array([[1,2], [1,3]])))
    [[ 3.86432845]
     [ 0.25877718]]
    >>> U, S, V = sci.svd([[1,2], [1,3]])
    >>> print(U, S, V)
    [[-0.57604844 -0.81741556]
     [-0.81741556  0.57604844]] [[ 3.86432845  0.        ]
     [ 0.          0.25877718]] [[-0.36059668 -0.93272184]
     [-0.93272184  0.36059668]]
    >>> # low level constructs
    >>> sci.run("y=ones(3,3)")
    >>> print(sci.get("y"))
    [[ 1.  1.  1.]
     [ 1.  1.  1.]
     [ 1.  1.  1.]]
    >>> sci.run("x=zeros(3,3)", verbose=True)
    >>> x = sci.call('rand', 1, 4)
    >>> print(x)
    [[  2.11324865e-01   7.56043854e-01   2.21134629e-04   3.30327092e-01]]
    >>> t = sci.call('rand', 1, 2, verbose=True)
    >>> y = np.zeros((3,3))
    >>> sci.put('y', y)
    >>> print(sci.get('y'))
    [[ 0.  0.  0.]
     [ 0.  0.  0.]
     [ 0.  0.  0.]]
    >>> from scilab2py import Struct
    >>> y = Struct()
    >>> y.b = 'spam'
    >>> y.c.d = 'eggs'
    >>> print(y.c['d'])
    eggs
    >>> print(y)
    {'c': {'d': 'eggs'}, 'b': 'spam'}
    >>> #getting help
    >>> sci.help('zeros')

.. image:: static/help.png

.. code-block:: python

    ********************
    DEMO COMPLETE!


