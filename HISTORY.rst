.. :changelog:

Release History
---------------

0.6.0 (2015-01-17)
++++++++++++++++++
- Add `convert_to_float` property that is True by default.
- Suppress output in dynamic function calls (using ';')


0.5.0 (2014-10-11)
++++++++++++++++++
- Make `eval` output match Octave session output.
  If verbose=True, print all Octave output.
  Return the last "ans" from Octave, if available.
  If you need the response, use `return_both` to get the
  `(resp, ans)` pair back
- As a result of the previous, Syntax Errors in Scilab code
  will now result in a closed session.
- Fix sizing of plots when in inline mode.
- Numerous corner case bug fixes.


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
