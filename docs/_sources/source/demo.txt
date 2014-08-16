
***********
Demo
***********

Output of Scliab2Py demo script, showing most of the features of the library.  Note that the two
plot commands will generate an interactive plot in the actual demo.
To run interactively:


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

    >>> sci.close()
    >>> xx = np.arange(-2*np.pi, 2*np.pi, 0.2)
    >>> sci.surf(np.subtract.outer(np.sin(xx), np.cos(xx)))
    Press Enter to continue...

.. image:: static/surf.png

.. code-block:: python

    >>> sci.close()
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
    >>> sci.eval("y=ones(3,3)")
    >>> print(sci.pull("y"))
    [[ 1.  1.  1.]
     [ 1.  1.  1.]
     [ 1.  1.  1.]]
    >>> sci.eval("x=zeros(3,3)", verbose=True)
    >>> t = sci.eval('rand(1, 2)')
    >>> y = np.zeros((3,3))
    >>> sci.push('y', y)
    >>> print(sci.pull('y'))
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
