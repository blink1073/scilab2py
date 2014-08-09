# -*- coding: utf-8 -*-
"""
===========
scilabmagic
===========

Magics for interacting with Scilab via Scilab2Py.

.. note::

  The ``scilab2py`` module needs to be installed separately and
  can be obtained using ``easy_install`` or ``pip``.

  You will also need a working copy of Scilab.

Usage
=====

To enable the magics below, execute ``%load_ext scilabmagic``.

``%scilab``

{SCILAB_DOC}

``%scilab_push``

{SCILAB_PUSH_DOC}

``%scilab_pull``

{SCILAB_PULL_DOC}

"""


#-----------------------------------------------------------------------------
#  Copyright (C) 2012 The IPython Development Team
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
#-----------------------------------------------------------------------------

import tempfile
from glob import glob
import os
from shutil import rmtree
import re

import numpy as np
import scilab2py
from xml.dom import minidom

from IPython.core.displaypub import publish_display_data
from IPython.core.magic import (Magics, magics_class, line_magic,
                                line_cell_magic, needs_local_scope)
from IPython.testing.skipdoctest import skip_doctest
from IPython.core.magic_arguments import (
    argument, magic_arguments, parse_argstring
)
from IPython.utils.py3compat import unicode_to_str
from IPython.utils.text import dedent


class ScilabMagicError(scilab2py.Scilab2PyError):
    pass

_mimetypes = {'png': 'image/png',
              'svg': 'image/svg+xml',
              'jpg': 'image/jpeg',
              'jpeg': 'image/jpeg'}


@magics_class
class ScilabMagics(Magics):
    """A set of magics useful for interactive work with Scilab via scilab2py.
    """
    def __init__(self, shell):
        """
        Parameters
        ----------
        shell : IPython shell

        """
        super(ScilabMagics, self).__init__(shell)
        self._sci = scilab2py.Scilab2Py()
        self._plot_format = 'png'

        # Allow publish_display_data to be overridden for
        # testing purposes.
        self._publish_display_data = publish_display_data

    @skip_doctest
    @line_magic
    def scilab_push(self, line):
        '''
        Line-level magic that pushes a variable to Scilab.

        `line` should be made up of whitespace separated variable names in the
        IPython namespace::

            In [7]: import numpy as np

            In [8]: X = np.arange(5)

            In [9]: X.mean()
            Out[9]: 2.0

            In [10]: %scilab_push X

            In [11]: %scilab mean(X)
            Out[11]: 2.0

        '''
        inputs = line.split(' ')
        for input in inputs:
            input = unicode_to_str(input)
            self._sci.put(input, self.shell.user_ns[input])

    @skip_doctest
    @line_magic
    def scilab_pull(self, line):
        '''
        Line-level magic that pulls a variable from Scilab.

        ::

            In [18]: _ = %scilab x = [1 2; 3 4]; y = 'hello'

            In [19]: %scilab_pull x y

            In [20]: x
            Out[20]:
            array([[ 1.,  2.],
                   [ 3.,  4.]])

            In [21]: y
            Out[21]: 'hello'

        '''
        outputs = line.split(' ')
        for output in outputs:
            output = unicode_to_str(output)
            self.shell.push({output: self._sci.get(output)})

    @skip_doctest
    @magic_arguments()
    @argument(
        '-i', '--input', action='append',
        help='Names of input variables to be pushed to Scilab. Multiple names '
             'can be passed, separated by commas with no whitespace.'
        )
    @argument(
        '-o', '--output', action='append',
        help='Names of variables to be pulled from Scilab after executing cell '
             'body. Multiple names can be passed, separated by commas with no '
             'whitespace.'
        )
    @argument(
        '-f', '--format', action='store',
        help='Plot format (png, svg or jpg).'
        )

    @needs_local_scope
    @argument(
        'code',
        nargs='*',
        )
    @line_cell_magic
    def scilab(self, line, cell=None, local_ns=None):
        '''
        Execute code in Scilab, and pull some of the results back into the
        Python namespace::

            In [9]: %scilab X = [1 2; 3 4]; mean(X)
            Out[9]: array([[ 2., 3.]])

        As a cell, this will run a block of Scilab code, without returning any
        value::

            In [10]: %%scilab
               ....: p = [-2, -1, 0, 1, 2]
               ....: polyout(p, 'x')

            -2*x^4 - 1*x^3 + 0*x^2 + 1*x^1 + 2

        In the notebook, plots are published as the output of the cell, e.g.::

            %scilab plot([1 2 3], [4 5 6])

        will create a line plot.

        Objects can be passed back and forth between Scilab and IPython via the
        -i and -o flags in line::

            In [14]: Z = np.array([1, 4, 5, 10])

            In [15]: %scilab -i Z mean(Z)
            Out[15]: array([ 5.])


            In [16]: %scilab -o W W = Z * mean(Z)
            Out[16]: array([  5.,  20.,  25.,  50.])

            In [17]: W
            Out[17]: array([  5.,  20.,  25.,  50.])

        The format of output plots can be specified::

            In [18]: %%scilab -f svg
                ...: plot([1, 2, 3]);

        '''
        # match current working directory
        self._sci.cd(os.getcwd())
        try:
            self._sci.getd('.')
        except scilab2py.Scilab2PyError:
            pass

        if self._first_call:
            # we need this var because `_` is a function in Scilab
            self._sci.put('last_magic_value', '')

        args = parse_argstring(self.scilab, line)

        # arguments 'code' in line are prepended to the cell lines
        if cell is None:
            code = ''
            return_output = True
        else:
            code = cell
            return_output = False

        code = ' '.join(args.code) + code

        # if there is no local namespace then default to an empty dict
        if local_ns is None:
            local_ns = {}

        if args.input:
            for input in ','.join(args.input).split(','):
                input = unicode_to_str(input)
                try:
                    val = local_ns[input]
                except KeyError:
                    val = self.shell.user_ns[input]
                self._sci.put(input, val)

        # generate plots in a temporary directory
        plot_dir = tempfile.mkdtemp().replace('\\', '/')

        if args.format is not None:
            plot_format = args.format
        else:
            plot_format = 'png'

        pre_call = '''
        h = gdf()
        h.figure_position = [0, 0]
        h.toolbar_visible = 'off'
        h.menubar_visible = 'off'
        h.infobar_visible = 'off'

        function handle_all_fig()
           ids_array=winsid();
           for i=1:length(ids_array)
              id=ids_array(i);
              outfile = sprintf('%(plot_dir)s/__ipy_sci_fig_%%03d', i);
              if '%(plot_format)s' == 'jpg' then
                xs2jpg(id, outfile + '.jpg')
              elseif '%(plot_format)s' == 'jpeg' then
                xs2jpg(id, outfile + '.jpeg')
              elseif '%(plot_format)s' == 'png' then
                xs2png(id, outfile)
              else
                xs2svg(id, outfile)
              end
              close(get_figure_handle(id));
           end
        endfunction
        ''' % locals()

        code = ' '.join((pre_call, code)).strip()

        try:
            resp = self._sci.run(code, verbose=False)
        except (scilab2py.Scilab2PyError) as exception:
            msg = str(exception)
            if 'Scilab Syntax Error' in msg:
                raise ScilabMagicError(msg)
            msg = msg.replace(pre_call.strip(), '')
            msg = re.sub('"""\s+', '"""\n', msg)
            msg = re.sub('\s+"""', '\n"""', msg)
            raise ScilabMagicError(msg)
        key = 'ScilabMagic.Scilab'
        display_data = []

        self._sci.eval_('handle_all_fig()')

        # Publish images
        images = []
        for imgfile in glob("%s/*" % plot_dir):
            with open(imgfile, 'rb') as fid:
                images.append(fid.read())
        rmtree(plot_dir)

        plot_mime_type = _mimetypes.get(plot_format, 'image/png')
        for image in images:
            display_data.append((key, {plot_mime_type: image}))

        if args.output:
            for output in ','.join(args.output).split(','):
                output = unicode_to_str(output)
                self.shell.push({output: self._sci.get(output)})

        for source, data in display_data:
            # source is deprecated in IPython 3.0.
            # specify with kwarg for backward compatibility.
            self._publish_display_data(source=source, data=data)

        if return_output:
            return resp
        elif not resp is None:
            print(resp)


__doc__ = __doc__.format(
    SCILAB_DOC=dedent(ScilabMagics.scilab.__doc__),
    SCILAB_PUSH_DOC=dedent(ScilabMagics.scilab_push.__doc__),
    SCILAB_PULL_DOC=dedent(ScilabMagics.scilab_pull.__doc__)
    )


def load_ipython_extension(ip):
    """Load the extension in IPython."""
    ip.register_magics(ScilabMagics)
