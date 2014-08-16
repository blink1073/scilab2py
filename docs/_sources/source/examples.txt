***********************
Examples
***********************

ScilabMagic
==========================
Scilab2Py provides a plugin for IPython to bring Scilab to the IPython prompt or the
IPython Notebook_.

.. _Notebook: http://nbviewer.ipython.org/github/blink1073/scilab2py/blob/master/example/scilabmagic_extension.ipynb?create=1


Script Examples
===============


Scripts in the directory where scilab2py was initialized, or those in the
Scilab path, can be called like any other Scilab function.
To explicitly add to the path, use::

   >>> from scilab2py import scilab
   >>> scilab.getd('/path/to/directory')

to add the directory in which your script is located to Scilab's path.


Roundtrip
---------

roundtrip.sci
+++++++++++++

::

    function [x, dtype] = roundtrip(y)

      // returns the variable it was given, and optionally the datatype

      x = y;

      if argn(1) == 2

             dtype = typeof(x);

      end

    endfunction


Python Session
++++++++++++++

.. code-block:: python

   >>> from scilab2py import scilab
   >>> import numpy as np
   >>> x = np.array([[1, 2], [3, 4]], dtype=float)
   >>> out, oclass = scilab.roundtrip(x)
   >>> import pprint
   >>> pprint.pprint([x, x.dtype, out, oclass, out.dtype])
  [array([[ 1.,  2.],
         [ 3.,  4.]]),
   dtype('float64'),
   array([[ 1.,  2.],
         [ 3.,  4.]]),
   u'constant',
   dtype('<f8')]



Test Datatypes
---------------

test_datatypes.sci
++++++++++++++++++

::

    function [data] = test_datatypes()
        // Test of returning a structure with multiple
        // nesting and multiple return types
        // Add a UTF char for test: 猫

        //////////////////////////////
        // numeric types
        // integers
        data.num.int.int8 = int8(-2^7);
        data.num.int.int16 = int16(-2^15);
        data.num.int.int32 = int32(-2^31);
        data.num.int.uint8 = uint8(2^8-1);
        data.num.int.uint16 = uint16(2^16-1);
        data.num.int.uint32 = uint32(2^32-1);

        // floats
        data.num.double = double(%pi);
        data.num.complex = complex(3, 1)
        data.num.complex_matrix = complex(1.2, 1.1) * eye(3, 3);

        // misc
        data.num.matrix = [1 2; 3 4];
        data.num.vector = [1 2 3 4];
        data.num.column_vector = [1;2;3;4];
        data.num.matrix3d = ones([2 3 4]) * %pi;


        //////////////////////////////
        // logical type
        //data.logical = [10 20 30 40 50] > 30;

        //////////////////////////////
        // string types
        data.string.basic = 'spam';

        //////////////////////////////
        // cell array types
        data.cell.array = {[0.4194 0.3629 -0.0000;
                            0.0376 0.3306 0.0000;
                            0 0 1.0000],
                           [0.5645 -0.2903 0;
                            0.0699 0.1855 0.0000;
                            0.8500 0.8250 1.0000]};

        //////////////////////////////
        // mixed struct
        data.mixed.array = [[1 2]; [3 4]];
        data.mixed.cell = {'1'};
        data.mixed.scalar = 1.8;

    endfunction


Python Session
+++++++++++++++

.. code-block:: python

   >>> from scilab2py import scilab
   >>> out = scilab.test_dataypes()
   >>> import pprint
   >>> pprint.pprint(out)
    {'cell': {'array': array([[ 0.4194,  0.3629, -0.    ],
           [ 0.0376,  0.3306,  0.    ],
           [ 0.    ,  0.    ,  1.    ],
           [ 0.5645, -0.2903,  0.    ],
           [ 0.0699,  0.1855,  0.    ],
           [ 0.85  ,  0.825 ,  1.    ]])},
     'mixed': {'array': array([[ 1.,  2.],
           [ 3.,  4.]]),
               'cell': u'1',
               'scalar': 1.8},
     'num': {'column_vector': array([[ 1.],
           [ 2.],
           [ 3.],
           [ 4.]]),
             'complex': (3+1j),
             'complex_matrix': array([[ 1.2+1.1j,  0.0+0.j ,  0.0+0.j ],
           [ 0.0+0.j ,  1.2+1.1j,  0.0+0.j ],
           [ 0.0+0.j ,  0.0+0.j ,  1.2+1.1j]]),
             'double': 3.1415926535897931,
             'int': {'int16': -32768,
                     'int32': -2147483648,
                     'int8': -128,
                     'uint16': 65535,
                     'uint32': 4294967295,
                     'uint8': 255},
             'matrix': array([[ 1.,  2.],
           [ 3.,  4.]]),
             'matrix3d': array([[ 3.14159265,  3.14159265,  3.14159265]]),
             'vector': array([[ 1.,  2.,  3.,  4.]])},
     'string': {'basic': u'spam'}}
