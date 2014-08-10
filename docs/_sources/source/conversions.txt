***********************
Conversions
***********************

Python to Scliab Types
----------------------

Shows the round-trip data types.

=============   ===========    =============
Python          Scilab         Python
=============   ===========    =============
int             double          np.float64
long            double          np.float64
float           double         np.float64
complex         double         np.complex128
str             string           unicode
unicode         stiring           unicode
bool            doubl          np.float64
None            double         np.float64
=============   ===========    =============

Numpy to Scilab Types
---------------------

Note that the errors are types that are not implemented.

=============   ===========    =============
Numpy           Scilab         Numpy
=============   ===========    =============
np.int8         double           np.float64
np.int16        double          np.float64
np.int32        double          np.float64
np.int64        double          np.float64
np.uint8        double          np.float64
np.uint16       double         np.float64
np.uint32       double         np.float64
np.uint64       double         np.float64
np.float16      ERROR          ERROR
np.float32      double         *np.float64*
np.float64      double         np.float64
np.float96      ERROR          ERROR
np.str          string           np.str
np.double       double         *np.float64*
np.complex64    double         *np.complex128*
np.complex128   double         np.complex128
np.complex192   ERROR          ERROR
np.object       cell           list
=============   ===========    =============

Python to Scilab Compound Types
-------------------------------

==================   ===========    ===============
Python               Scilab         Python
==================   ===========    ===============
list of strings      cell (1-d)     list of strings
list of mixed type   ERROR           list of mixed type
nested string list   ERROR           list of strings
tuple of strings     cell           list of strings
nested dict          struct         Struct
set of int32         double          np.float64
==================   ===========    ===============

Scilab to Python Types
----------------------

These are the unique values apart from the Python to Scilab lists.

===============  =================
Scilab           Python
===============  =================
matrix           ndarray
cell (2-d)       ERROR
cell (scalar)    scalar
cell array       ERROR
struct           Struct
struct (nested)  Struct (nested)
struct array     ERROR
logical          ERROR
===============  =================

