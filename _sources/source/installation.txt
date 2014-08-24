Installation
************************

Library Installation
--------------------
You must have Scilab_ 5.5 installed and in your PATH (see instructions below).
Additionally, you must have the Numpy and Scipy libraries installed.  On Windows, you can get the install files here_.

The best way to install this library is by using pip_::

   pip install scilab2py


.. _Scilab: http://www.scilab.org/download/
.. _here: http://scipy.org/Download
.. _pip: http://www.pip-installer.org/en/latest/installing.html


Scilab Path Installation
-----------------------------

The goal is to be able to call Scilab from the command prompt.

On Posix systems, open a shell and type ``scilab --nw``.
If that does not work, add the path to your Scilab executable in your PATH in your ``.profile`` file.

On Windows, open a ``cmd`` prompt and type ``Scilex``.
If that does not work, you need to find your ``scilab-x.x.x\bin`` directory (probably in ``C:/Program Files``) and add that to your path.
You can do so from the Environmental Variables dialog for your version of Windows, or set from the command prompt::

      setx PATH "%PATH%;<path-to-scilab-bin-dir>"
