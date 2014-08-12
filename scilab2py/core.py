"""
.. module:: session
   :synopsis: Main module for scilabpy package.
              Contains the Scilab session object Scilab2Py

.. moduleauthor:: Steven Silvester <steven.silvester@ieee.org>

"""
from __future__ import print_function
import os
import re
import atexit
import doctest
import signal
import subprocess
import sys
import threading
import time

from scilab2py.matwrite import MatWrite
from scilab2py.matread import MatRead
from scilab2py.utils import (
    get_nout, Scilab2PyError, get_log, Struct, _remove_temp_files)
from scilab2py.compat import PY2, queue, unicode


class Scilab2Py(object):

    """Manages a Scliab session.

    Uses MAT files to pass data between Scilab and Numpy.
    The function must either exist as an file in this directory or
    on Scilab's path.

    You may provide a logger object for logging events, or the scilab.get_log()
    default will be used.  Events will be logged as debug unless verbose is set
    when calling a command, then they will be logged as info.

    Parameters
    ----------
    logger : logging object, optional
        Optional logger to use for Scilab2Py session
    timeout : float, opional
        Timeout in seconds for commands
    oned_as : {'row', 'column'}, optional
        If 'column', write 1-D numpy arrays as column vectors.
        If 'row', write 1-D numpy arrays as row vectors.}
    temp_dir : str, optional
        If specified, the session's MAT files will be created in the
        directory, otherwise a default directory is used.  This can be
        a shared memory (tmpfs) path.
    """

    def __init__(self, logger=None, timeout=-1, oned_as='row',
                 temp_dir=None):
        """Start Scilab and create our MAT helpers
        """
        self._oned_as = oned_as
        self._temp_dir = temp_dir
        atexit.register(lambda: _remove_temp_files(temp_dir))
        self.timeout = timeout
        if not logger is None:
            self.logger = logger
        else:
            self.logger = get_log()
        self._session = None
        self.restart()

    def __enter__(self):
        """Return Scilab object, restart session if necessary"""
        if not self._session:
            self.restart()
        return self

    def __exit__(self, type, value, traceback):
        """Close session"""
        self.close()

    def close(self):
        """Closes this Scilab session and removes temp files
        """
        if self._session:
            self._session.close()
        self._session = None
        try:
            self._writer.remove_file()
            self._reader.remove_file()
        except Scilab2PyError:
            pass

    def run(self, script, **kwargs):
        """
        Run artibrary Scilab code.

        Parameters
        -----------
        script : str
            Command script to send to Scilab for execution.
        verbose : bool, optional
            Log Scilab output at info level.

        Returns
        -------
        out : str
            Scilab printed output.

        Raises
        ------
        Scilab2PyError
            If the script cannot be run by Scilab.

        Examples
        --------
        >>> from scilab2py import scilab
        >>> scilab.run('y=ones(3,3)')
        >>> print(scilab.get('y'))
        [[ 1.  1.  1.]
         [ 1.  1.  1.]
         [ 1.  1.  1.]]
        >>> scilab.run('x = mean([[1, 2], [3, 4]])')
        array([[ 1.,  1.,  1.],
               [ 1.,  1.,  1.],
               [ 1.,  1.,  1.]])

        """
        # don't return a value from a script
        kwargs['nout'] = 0
        return self.call(script, **kwargs)

    def call(self, func, *inputs, **kwargs):
        """
        Call an Scilab function with optional arguments.

        Parameters
        ----------
        func : str
            Function name to call.
        inputs : array_like
            Variables to pass to the function.
        nout : int, optional
            Number of output arguments.
            This is set automatically based on the number of
            return values requested (see example below).
            You can override this behavior by passing a
            different value.
        verbose : bool, optional
             Log Scilab output at info level.

        Returns
        -------
        out : str or tuple
            If nout > 0, returns the values from Scilab as a tuple.
            Otherwise, returns the output displayed by Scilab.

        Raises
        ------
        Scilab2PyError
            If the call is unsucessful.

        Examples
        --------
        >>> from scilab2py import scilab
        >>> b = scilab.call('ones', 1, 2)
        >>> print(b)
        [[ 1.  1.]]
        >>> x, y = 1, 2
        >>> a = scilab.call('zeros', x, y)
        >>> a
        array([[ 0.,  0.]])
        >>> U, S, V = scilab.call('svd', [[1, 2], [1, 3]])
        >>> print((U, S, V))
        (array([[-0.57604844, -0.81741556],
               [-0.81741556,  0.57604844]]), array([[ 3.86432845,  0.        ],
               [ 0.        ,  0.25877718]]), array([[-0.36059668, -0.93272184],
               [-0.93272184,  0.36059668]]))

        """
        verbose = kwargs.get('verbose', False)
        nout = kwargs.get('nout', get_nout())
        timeout = kwargs.get('timeout', self.timeout)
        argout_list = ['ans']
        if '=' in func:
            nout = 0

        # handle references to script names - and paths to them
        if func.endswith('.sci'):
            if os.path.dirname(func):
                self.getd(os.path.dirname(func))
                func = os.path.basename(func)
            else:
                try:
                    self.getd('.')
                except Scilab2PyError:
                    pass
            func = func[:func.index('.')]

        # these three lines will form the commands sent to Scilab
        # load("-v6", "infile", "invar1", ...)
        # [a, b, c] = foo(A, B, C)
        # save("-v6", "outfile", "outvar1", ...)
        load_line = call_line = save_line = ''

        if nout:
            # create a dummy list of var names ("a", "b", "c", ...)
            # use ascii char codes so we can increment
            argout_list, save_line = self._reader.setup(nout)
            call_line = '[{0}] = '.format(', '.join(argout_list))
        if inputs:
            argin_list, load_line = self._writer.create_file(inputs)
            call_line += '{0}({1})'.format(func, ', '.join(argin_list))
        elif nout and not '(' in func:
            # call foo() - no arguments
            call_line += '{0}()'.format(func)
        else:
            # run foo
            call_line += '{0}'.format(func)

        if not call_line.endswith(')') and nout:
            call_line += '()'
        else:
            call_line += ';'

        # create the command and execute in Scilab
        cmd = [load_line, call_line, save_line]
        data = self._eval(cmd, verbose=verbose, timeout=timeout)
        if isinstance(data, dict) and not isinstance(data, Struct):
            data = [data.get(v, None) for v in argout_list]
            if len(data) == 1 and data[0] is None:
                data = None
        if verbose:
            self.logger.info(data)
        else:
            self.logger.debug(data)
        return data

    def put(self, names, var, verbose=False, timeout=-1):
        """
        Put a variable into the Scilab session.

        Parameters
        ----------
        names : str or list
            Name of the variable(s).
        var : object or list
            The value(s) to pass.
        timeout : float
            Time to wait for response from Scilab (per character).

        Examples
        --------
        >>> from scilab2py import scilab
        >>> y = [1, 2]
        >>> scilab.put('y', y)
        >>> scilab.get('y')
        array([[ 1.,  2.]])
        >>> scilab.put(['x', 'y'], ['spam', [1., 2., 3., 4.]])
        >>> scilab.get(['x', 'y'])
        [u'spam', array([[ 1.,  2.,  3.,  4.]])]

        """
        if isinstance(names, (str, unicode)):
            var = [var]
            names = [names]
        for name in names:
            if name.startswith('_'):
                raise Scilab2PyError('Invalid name {0}'.format(name))
        _, load_line = self._writer.create_file(var, names)
        self._eval(load_line, verbose=verbose, timeout=timeout)

    def get(self, var, verbose=False, timeout=-1):
        """
        Retrieve a value from the Scilab session.

        Parameters
        ----------
        var : str
            Name of the variable to retrieve.
        timeout : float
            Time to wait for response from Scilab (per character).

        Returns
        -------
        out : object
            Object returned by Scilab.

        Raises:
          Scilab2PyError
            If the variable does not exist in the Scilab session.

        Examples:
          >>> from scilab2py import scilab
          >>> y = [1, 2]
          >>> scilab.put('y', y)
          >>> scilab.get('y')
          array([[ 1.,  2.]])
          >>> scilab.put(['x', 'y'], ['spam', [1, 2, 3, 4]])
          >>> scilab.get(['x', 'y'])
          [u'spam', array([[ 1.,  2.,  3.,  4.]])]

        """
        if isinstance(var, (str, unicode)):
            var = [var]
        argout_list, save_line = self._reader.setup(len(var), var)
        data = self._eval(save_line, verbose=verbose, timeout=timeout)
        if isinstance(data, dict) and not isinstance(data, Struct):
            return [data.get(v, None) for v in argout_list]
        else:
            return data

    def _eval(self, cmds, verbose=True, log=True, timeout=-1):
        """
        Perform raw Scilab command.

        This is a low-level command, and should not technically be used
        directly.  The API could change. You have been warned.

        Parameters
        ----------
        cmds : str or list
            Commands(s) to pass directly to Scilab.
        verbose : bool, optional
             Log Scilab output at info level.
        timeout : float
            Time to wait for response from Scilab (per character).

        Returns
        -------
        out : str
            Results printed by Scilab.

        Raises
        ------
        Scilab2PyError
            If the command(s) fail.

        """
        if not self._session:
            raise Scilab2PyError('No Scilab Session')

        if isinstance(cmds, (str, unicode)):
            cmds = [cmds]

        if verbose and log:
            [self.logger.info(line) for line in cmds]

        elif log:
            [self.logger.debug(line) for line in cmds]

        if timeout == -1:
            timeout = self.timeout

        post_call = ''
        for cmd in cmds:

            match = re.match('([a-z][a-zA-Z0-9_]*) *=', cmd)
            if match and not cmd.strip().endswith(';'):
                post_call = 'ans = %s' % match.groups()[0]
                break

            match = re.match('([a-z][a-zA-Z0-9_]*)\Z', cmd.strip())
            if match and not cmd.strip().endswith(';'):
                post_call = 'ans = %s' % match.groups()[0]
                break

        cmds.append(post_call)

        try:
            resp = self._session.evaluate(cmds, verbose, log, self.logger,
                                          timeout=timeout)
        except KeyboardInterrupt:
            self._session.interrupt()
            return 'Scilab Session Interrupted'

        if os.path.exists(self._reader.out_file):
            try:
                return self._reader.extract_file()
            except (TypeError, IOError):
                pass

        if resp:
            return resp

    def _make_scilab_command(self, name, doc=None):
        """Create a wrapper to an Scilab procedure or object

        Adapted from the mlabwrap project

        """
        def scilab_command(*args, **kwargs):
            """ Scilab command """
            kwargs['nout'] = get_nout()
            if name == 'getd':
                kwargs['nout'] = 0
            kwargs['verbose'] = kwargs.get('verbose', False)
            return self.call(name, *args, **kwargs)

        # convert to ascii for pydoc
        try:
            doc = doc.encode('ascii', 'replace').decode('ascii')
        except UnicodeDecodeError:
            pass

        scilab_command.__doc__ = "\n" + doc
        scilab_command.__name__ = name

        return scilab_command

    def _get_doc(self, name):
        """
        Get the documentation of an Scilab procedure or object.

        Parameters
        ----------
        name : str
            Function name to search for.

        Returns
        -------
        out : str
          Documentation string.

        Raises
        ------
        Scilab2PyError
           If the procedure or object does not exist.

        """
        doc = "No documentation available for `%s`" % name

        try:
            typeof = self._eval('typeof(%s);' % name)

        except Scilab2PyError:
            raise Scilab2PyError('No function named `%s`' % name)

        if typeof == 'fptr':
            doc = "`%s` is a built-in Scilab function." % name
            doc += """\nUse run("help %s") for full docs.""" % name

        elif typeof == 'function':
            lines = self._eval('fun2string(%s);' % name)
            lines = lines.replace('!', ' ').splitlines()

            docs = [lines[0].replace('ans(', '%s(' % name), ' ']

            in_doc = False
            for line in lines[1:]:
                line = line.strip()

                if line.startswith('//'):
                    docs.append(line[2:])
                    in_doc = True

                elif in_doc and line:
                    break

            doc = '\n'.join(docs)

        else:
            raise Scilab2PyError('No function named `%s`' % name)

        return doc

    def __getattr__(self, attr):
        """Automatically creates a wapper to an Scilab function or object.

        Adapted from the mlabwrap project.

        """
        # needed for help(Scilab2Py())
        if attr == '__name__':
            return super(Scilab2Py, self).__getattr__(attr)
        elif attr == '__file__':
            return __file__
        # close_ -> close
        if attr[-1] == "_":
            name = attr[:-1]
        else:
            name = attr
        doc = self._get_doc(name)
        scilab_command = self._make_scilab_command(name, doc)
        #!!! attr, *not* name, because we might have python keyword name!
        setattr(self, attr, scilab_command)
        return scilab_command

    def restart(self):
        """Restart an Scilab session in a clean state
        """
        self._reader = MatRead(self._temp_dir)
        self._writer = MatWrite(self._temp_dir, self._oned_as)
        self._session = _Session(self._reader.out_file)


class _Reader(object):

    """Read characters from an Scilab session in a thread.
    """

    def __init__(self, fid, queue):
        self.fid = fid
        self.queue = queue
        self.thread = threading.Thread(target=self.read_incoming)
        self.thread.setDaemon(True)
        self.thread.start()

    def read_incoming(self):
        """"Read text a chunk at a time, parsing into lines
        and putting them in the queue.
        If there is a line with only a ">" char, put that on the queue
        """
        buf = ''
        while 1:
            try:
                buf += os.read(self.fid, 1024).decode('utf8')
            except:
                self.queue.put(None)
                return
            lines = buf.splitlines()
            for line in lines[:-1]:
                self.queue.put(line)
            if buf.endswith('\n'):
                self.queue.put(lines[-1])
                buf = ''
            else:
                buf = lines[-1]


class _Session(object):

    """Low-level session Scilab session interaction.
    """

    def __init__(self, outfile):
        self.timeout = int(1e6)
        self.read_queue = queue.Queue()
        self.proc = self.start()
        self._first = True
        self.stdout = sys.stdout
        self.outfile = outfile
        self.set_timeout()
        atexit.register(self.close)

    def start(self):
        """
        Start an scilab session in a subprocess.

        Returns
        =======
        out : fid
            File descriptor for the Scilab subprocess

        Raises
        ======
        Scilab2PyError
            If the session is not opened sucessfully.

        Notes
        =====
        Options sent to Scilab: -q is quiet startup, --braindead is
        Matlab compatibilty mode.

        """
        return self.start_subprocess()

    def start_subprocess(self):
        """Start scilab using a subprocess (no tty support)"""
        errmsg = ('\n\nPlease install Scilab and put it in your path\n')
        ON_POSIX = 'posix' in sys.builtin_module_names
        self.rfid, wpipe = os.pipe()
        rpipe, self.wfid = os.pipe()
        kwargs = dict(close_fds=ON_POSIX, bufsize=0, stdin=rpipe,
                      stderr=wpipe, stdout=wpipe)
        proc_name = 'scilab'
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            kwargs['startupinfo'] = startupinfo
            proc_name = 'Scilex'
        try:
            proc = subprocess.Popen([proc_name, '-nw'],
                                    **kwargs)
        except OSError:  # pragma: no cover
            raise Scilab2PyError(errmsg)
        else:
            self.reader = _Reader(self.rfid, self.read_queue)
            return proc

    def set_timeout(self, timeout=-1):
        if timeout == -1:
            timeout = int(1e6)
        self.timeout = timeout

    def evaluate(self, cmds, verbose=True, log=True, logger=None, timeout=-1):
        """Perform the low-level interaction with an Scilab Session
        """
        if not timeout == -1:
            self.set_timeout(timeout)

        if not self.proc:
            raise Scilab2PyError('Session Closed, try a restart()')

        if self._first:
            self.write('try;getd(".");catch;end\n')
            self._first = False

        if os.path.exists(self.outfile):
            try:
                os.remove(self.outfile)
            except OSError:
                pass

        # use ascii code 2 for start of text, 3 for end of text, and
        # 24 to signal an error
        exprs = []
        for cmd in cmds:
            items = cmd.replace(';', '\n').split('\n')
            exprs.extend([i.strip() for i in items
                          if i.strip()
                          and not i.strip().startswith(('%', '#'))])

        expr = '\n'.join(exprs)
        expr = expr.replace('"', '""')
        expr = expr.replace("'", "''")
        expr = expr.replace('\n', ';')

        output = """
        clear("ans");
        disp(char(2));
        if execstr("%s", "errcatch") <> 0 then
            disp(lasterror())
            disp(char(24))
        else
            if exists("ans") then
                last_ans = ans;
                if type(last_ans) == 4 then
                    last_ans = double(last_ans)
                end
                if or(type(last_ans) == [1,2,3,5,6,7,8,10]) then
                    _ = last_ans;
                    if exists("a__") == 0 then
                        try
                            savematfile -v6 %s _;
                        catch
                            disp(_)
                        end
                    end
                elseif type(last_ans)
                    disp(last_ans);
                end
                clear("last_ans")
            end
            disp(char(3))
        end""" % (expr, self.outfile)

        if len(cmds) == 5:
            main_line = cmds[2].strip()
        else:
            main_line = '\n'.join(cmds)

        self.write(output + '\n')
        self.expect(chr(2))

        debug_prompt = ("Type 'resume' or 'abort' to return to "
                        "standard level prompt.")

        resp = []
        while 1:
            line = self.readline()

            if chr(3) in line:
                break

            elif chr(24) in line:
                msg = ('Scilab2Py tried to run:\n"""\n{0}\n"""\n'
                       'Scilab returned:\n{1}'
                       .format(main_line, '\n'.join(resp)))
                raise Scilab2PyError(msg)

            elif line.strip() == debug_prompt:
                self.interact('-1->')

            if verbose and logger:
                logger.info(line)

            elif log and logger:
                logger.debug(line)

            if resp or line.strip():
                resp.append(line)

        return '\n'.join(resp).rstrip()

    def interrupt(self):
        self.proc.send_signal(signal.SIGINT)

    def expect(self, strings):
        """Look for a string or strings in the incoming data"""
        if not isinstance(strings, list):
            strings = [strings]
        lines = []
        while 1:
            line = self.readline()
            lines.append(line)
            if line:
                for string in strings:
                    if re.search(string, line):
                        return '\n'.join(lines)

    def readline(self):
        t0 = time.time()
        while 1:
            try:
                val = self.read_queue.get_nowait()
            except queue.Empty:
                pass
            else:
                if val is None:
                    self.close()
                    return
                else:
                    return val
            time.sleep(1e-6)
            if (time.time() - t0) > self.timeout:
                self.close()
                raise Scilab2PyError('Session Timed Out, closing')

    def write(self, message):
        """Write a message to the process using utf-8 encoding"""
        os.write(self.wfid, message.encode('utf-8'))

    def interact(self, prompt='- 1->'):
        """Manage an Scilab Debug Prompt interaction"""
        msg = 'Entering Scilab Debug Prompt...\n%s' % prompt
        self.stdout.write(msg)
        while 1:
            inp_func = input if not PY2 else raw_input
            try:
                inp = inp_func()
            except EOFError:
                return
            if inp in ['exit', 'quit', 'resume', 'abort',
                       'exit()', 'quit()']:
                inp = 'resume'
            msg = 'disp(char(2));' + inp + '\ndisp(char(3))\n'
            self.write(msg)
            if inp == 'resume':
                self.write('resume\n')
                self.write('clear _\n')
                return
            self.expect(chr(2))
            output = self.expect(chr(3))
            output = output[:output.index(chr(3))].rstrip()
            self.stdout.write(output + '\n\n')
            self.stdout.write(prompt)

    def close(self):
        """Cleanly close an Scilab session
        """
        try:
            self.proc.terminate()
        except (OSError, AttributeError):  # pragma: no cover
            pass
        self.proc = None

    def __del__(self):
        try:
            self.proc.terminate()
        except (OSError, AttributeError):
            pass


def _test():  # pragma: no cover
    """Run the doctests for this module.
    """
    print('Starting doctest')
    doctest.testmod()
    print('Completed doctest')


if __name__ == "__main__":  # pragma: no cover
    _test()
