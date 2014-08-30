.. :changelog:

Release History
---------------

0.4.0 (2014-08-30)
++++++++++++++++++
- Add suppport for Scilab 5.4


0.3.0 (2014-08-23)
++++++++++++++++++
- Allow keyword arguments in functions: `scilab.plot([1,2,3], linewidth=2))`
  These are translated to ("prop", value) arguments to the function.
- Add option to show plotting gui with `-g` flag in ScilabMagic.
- Add ability to specify the Scilab executable as a keyword argument to
  the Scilab2Py object.
 - Add specifications for plot saving instead of displaying plots to `eval` and
    dynamic functions.


0.2.0 (2014-08-14)
++++++++++++++++++
- Streamline API to mirror Oct2Py 2.0.0
- Python 3 support
- Bug fixes and usability improvements.


0.1.0 (2014-08-12)
++++++++++++++++++

- Initial Release
