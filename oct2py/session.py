"""
.. module:: session
   :synopsis: Main module for oct2py package.
              Contains the Octave session object Oct2Py

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

from .matwrite import MatWrite
from .matread import MatRead
from .utils import get_nout, Oct2PyError, get_log
from .compat import unicode, PY2, queue


# TODO: do NOT call for help when creating a file
#       HOW do we handling inline plotting?

class Oct2Py(object):

    """Manages an Octave session.

    Uses MAT files to pass data between Octave and Numpy.
    The function must either exist as an m-file in this directory or
    on Octave's path.
    The first command will take about 0.5s for Octave to load up.
    The subsequent commands will be much faster.

    You may provide a logger object for logging events, or the oct2py.get_log()
    default will be used.  Events will be logged as debug unless verbose is set
    when calling a command, then they will be logged as info.

    Parameters
    ----------
    logger : logging object, optional
        Optional logger to use for Oct2Py session
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
        """Start Octave and create our MAT helpers
        """
        self._oned_as = oned_as
        self._temp_dir = temp_dir
        self.timeout = timeout
        if not logger is None:
            self.logger = logger
        else:
            self.logger = get_log()
        self._session = None
        self.restart()

    def __enter__(self):
        """Return octave object, restart session if necessary"""
        if not self._session:
            self.restart()
        return self

    def __exit__(self, type, value, traceback):
        """Close session"""
        self.close()

    def close(self):
        """Closes this octave session and removes temp files
        """
        if self._session:
            self._session.close()
        self._session = None
        try:
            self._writer.remove_file()
            self._reader.remove_file()
        except Oct2PyError:
            pass

    def run(self, script, **kwargs):
        """
        Run artibrary Octave code.

        Parameters
        -----------
        script : str
            Command script to send to Octave for execution.
        verbose : bool, optional
            Log Octave output at info level.

        Returns
        -------
        out : str
            Octave printed output.

        Raises
        ------
        Oct2PyError
            If the script cannot be run by Octave.

        Examples
        --------
        >>> from oct2py import octave
        >>> out = octave.run('y=ones(3,3)')
        >>> print(out)
        y =
        <BLANKLINE>
                1        1        1
                1        1        1
                1        1        1
        <BLANKLINE>
        >>> octave.run('x = mean([[1, 2], [3, 4]])')
        u'x =  2.5000'

        """
        # don't return a value from a script
        kwargs['nout'] = 0
        return self.call(script, **kwargs)

    def call(self, func, *inputs, **kwargs):
        """
        Call an Octave function with optional arguments.

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
             Log Octave output at info level.

        Returns
        -------
        out : str or tuple
            If nout > 0, returns the values from Octave as a tuple.
            Otherwise, returns the output displayed by Octave.

        Raises
        ------
        Oct2PyError
            If the call is unsucessful.

        Examples
        --------
        >>> from oct2py import octave
        >>> b = octave.call('ones', 1, 2)
        >>> print(b)
        [[ 1.  1.]]
        >>> x, y = 1, 2
        >>> a = octave.call('zeros', x, y)
        >>> a
        array([[ 0.,  0.]])
        >>> U, S, V = octave.call('svd', [[1, 2], [1, 3]])
        >>> print((U, S, V))
        (array([[-0.57604844, -0.81741556],
               [-0.81741556,  0.57604844]]), array([[ 3.86432845,  0.        ],
               [ 0.        ,  0.25877718]]), array([[-0.36059668, -0.93272184],
               [-0.93272184,  0.36059668]]))

        """
        verbose = kwargs.get('verbose', False)
        nout = kwargs.get('nout', get_nout())
        timeout = kwargs.get('timeout', self.timeout)

        # handle references to script names - and paths to them
        if func.endswith('.sce'):
            if os.path.dirname(func):
                self.addpath(os.path.dirname(func))
                func = os.path.basename(func)
            func = func[:-2]

        # these three lines will form the commands sent to Octave
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
        elif nout:
            # call foo() - no arguments
            call_line += '{0}()'.format(func)
        else:
            # run foo
            call_line += '{0}'.format(func)

        pre_call = '\n__oct2py_figures = [];\n'
        post_call = ''

        if 'command' in kwargs and not '__ipy_figures' in func:
            if not call_line.endswith(')') and nout:
                call_line += '();\n'
            else:
                call_line += ';\n'
            post_call += """
            # Save output of the last execution
                if exists("ans") == 1
                  _ = ans;
                else
                  _ = "__no_answer";
                end
            """

        # do not interfere with octavemagic logic
        if not "DefaultFigureCreateFcn" in call_line:
            post_call += """
            if exists("__oct2py_figures")
                for f = __oct2py_figures
                    try
                       refresh(f);
                    end
                end
            end"""

        # create the command and execute in octave
        cmd = [load_line, pre_call, call_line, post_call, save_line]
        resp = self._eval(cmd, verbose=verbose, timeout=timeout)

        if nout:
            return self._reader.extract_file(argout_list)
        elif 'command' in kwargs:
            try:
                ans = self.get('_')
            except (KeyError, Oct2PyError):
                return
            # Unfortunately, Octave doesn't have a "None" object,
            # so we can't return any NaN outputs
            if isinstance(ans, (str, unicode)) and ans == "__no_answer":
                ans = None
            return ans
        else:
            return resp

    def put(self, names, var, verbose=False, timeout=-1):
        """
        Put a variable into the Octave session.

        Parameters
        ----------
        names : str or list
            Name of the variable(s).
        var : object or list
            The value(s) to pass.
        timeout : float
            Time to wait for response from Octave (per character).

        Examples
        --------
        >>> from oct2py import octave
        >>> y = [1, 2]
        >>> octave.put('y', y)
        >>> octave.get('y')
        array([[1, 2]])
        >>> octave.put(['x', 'y'], ['spam', [1, 2, 3, 4]])
        >>> octave.get(['x', 'y'])
        (u'spam', array([[1, 2, 3, 4]]))

        """
        if isinstance(names, str):
            var = [var]
            names = [names]
        for name in names:
            if name.startswith('_'):
                raise Oct2PyError('Invalid name {0}'.format(name))
        _, load_line = self._writer.create_file(var, names)
        self._eval(load_line, verbose=verbose, timeout=timeout)

    def get(self, var, verbose=False, timeout=-1):
        """
        Retrieve a value from the Octave session.

        Parameters
        ----------
        var : str
            Name of the variable to retrieve.
        timeout : float
            Time to wait for response from Octave (per character).

        Returns
        -------
        out : object
            Object returned by Octave.

        Raises:
          Oct2PyError
            If the variable does not exist in the Octave session.

        Examples:
          >>> from oct2py import octave
          >>> y = [1, 2]
          >>> octave.put('y', y)
          >>> octave.get('y')
          array([[1, 2]])
          >>> octave.put(['x', 'y'], ['spam', [1, 2, 3, 4]])
          >>> octave.get(['x', 'y'])
          (u'spam', array([[1, 2, 3, 4]]))

        """
        if isinstance(var, str):
            var = [var]
        # make sure the variable(s) exist
        for variable in var:
            if self._eval('exists "{0}"'.format(variable),
                          verbose=False) == 'ans = 0' and not variable == '_':
                raise Oct2PyError('{0} does not exist'.format(variable))
        argout_list, save_line = self._reader.setup(len(var), var)
        self._eval(save_line, verbose=verbose, timeout=timeout)
        return self._reader.extract_file(argout_list)

    def lookfor(self, string, verbose=False, timeout=-1):
        """
        Call the Octave "lookfor" command.

        Uses with the "-all" switch to search within help strings.

        Parameters
        ----------
        string : str
            Search string for the lookfor command.
        verbose : bool, optional
             Log Octave output at info level.
        timeout : float
            Time to wait for response from Octave (per character).

        Returns
        -------
        out : str
            Output from the Octave lookfor command.

        """
        return self.run('lookfor -all {0}'.format(string), verbose=verbose,
                        timeout=timeout)

    def _eval(self, cmds, verbose=True, log=True, timeout=-1):
        """
        Perform raw Octave command.

        This is a low-level command, and should not technically be used
        directly.  The API could change. You have been warned.

        Parameters
        ----------
        cmds : str or list
            Commands(s) to pass directly to Octave.
        verbose : bool, optional
             Log Octave output at info level.
        timeout : float
            Time to wait for response from Octave (per character).

        Returns
        -------
        out : str
            Results printed by Octave.

        Raises
        ------
        Oct2PyError
            If the command(s) fail.

        """
        if not self._session:
            raise Oct2PyError('No Octave Session')
        if isinstance(cmds, str):
            cmds = [cmds]
        if verbose and log:
            [self.logger.info(line) for line in cmds]
        elif log:
            [self.logger.debug(line) for line in cmds]
        if timeout == -1:
            timeout = self.timeout
        try:
            return self._session.evaluate(cmds, verbose, log, self.logger,
                                          timeout=timeout)
        except KeyboardInterrupt:
            self._session.interrupt()
            return 'Octave Session Interrupted'

    def _make_octave_command(self, name, doc=None):
        """Create a wrapper to an Octave procedure or object

        Adapted from the mlabwrap project

        """
        def octave_command(*args, **kwargs):
            """ Octave command """
            kwargs['nout'] = get_nout()
            kwargs['verbose'] = kwargs.get('verbose', False)
            if not 'Built-in Function' in doc:
                self._eval('clear {0}'.format(name), log=False, verbose=False)
            kwargs['command'] = True
            return self.call(name, *args, **kwargs)
        # convert to ascii for pydoc
        try:
            doc = doc.encode('ascii', 'replace').decode('ascii')
        except UnicodeDecodeError:
            pass
        octave_command.__doc__ = "\n" + doc
        octave_command.__name__ = name
        return octave_command

    def _get_doc(self, name):
        """
        Get the documentation of an Octave procedure or object.

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
        Oct2PyError
           If the procedure or object does not exist.

        """
        if name == 'keyboard':
            return 'Built-in Function: keyboard ()'
        exist = self._eval('exists "{0}"'.format(name), log=False, verbose=False)
        if exist.strip() == 'ans = 0':
            msg = 'Name: "%s" does not exist on the Octave session path'
            raise Oct2PyError(msg % name)
        doc = 'No documentation for %s' % name
        try:
            doc = self._eval('help {0}'.format(name), log=False, verbose=False)
        except Oct2PyError as e:
            if 'syntax error' in str(e):
                raise(e)
            try:
                doc = self._eval('type {0}'.format(name), log=False,
                                 verbose=False)
                doc = '\n'.join(doc.splitlines()[:3])
            except Oct2PyError as e:
                pass
        return doc

    def __getattr__(self, attr):
        """Automatically creates a wapper to an Octave function or object.

        Adapted from the mlabwrap project.

        """
        # needed for help(Oct2Py())
        if attr == '__name__':
            return super(Oct2Py, self).__getattr__(attr)
        elif attr == '__file__':
            return __file__
        # close_ -> close
        if attr[-1] == "_":
            name = attr[:-1]
        else:
            name = attr
        doc = self._get_doc(name)
        octave_command = self._make_octave_command(name, doc)
        #!!! attr, *not* name, because we might have python keyword name!
        setattr(self, attr, octave_command)
        return octave_command

    def restart(self):
        """Restart an Octave session in a clean state
        """
        self._session = _Session()
        self._reader = MatRead(self._temp_dir)
        self._writer = MatWrite(self._temp_dir, self._oned_as)

    def __del__(self):
        try:
            self.close()
        except AttributeError:
            pass


class _Reader(object):

    """Read characters from an Octave session in a thread.
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
        debug_prompt = re.compile(r'\A[\w]+>>? ')
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
            elif re.match(debug_prompt, lines[-1]):
                self.queue.put(lines[-1])
                buf = ''
            else:
                buf = lines[-1]


class _Session(object):

    """Low-level session Octave session interaction.
    """

    def __init__(self):
        self.timeout = int(1e6)
        self.read_queue = queue.Queue()
        self.proc = self.start()
        self.stdout = sys.stdout
        self.set_timeout()
        atexit.register(self.close)

    def start(self):
        """
        Start an octave session in a subprocess.

        Returns
        =======
        out : fid
            File descriptor for the Octave subprocess

        Raises
        ======
        Oct2PyError
            If the session is not opened sucessfully.

        Notes
        =====
        Options sent to Octave: -q is quiet startup, --braindead is
        Matlab compatibilty mode.

        """
        return self.start_subprocess()

    def start_subprocess(self):
        """Start octave using a subprocess (no tty support)"""
        errmsg = ('\n\nPlease install GNU Octave and put it in your path\n')
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
            raise Oct2PyError(errmsg)
        else:
            self.reader = _Reader(self.rfid, self.read_queue)
            return proc

    def set_timeout(self, timeout=-1):
        if timeout == -1:
            timeout = int(1e6)
        self.timeout = timeout

    def evaluate(self, cmds, verbose=True, log=True, logger=None, timeout=-1):
        """Perform the low-level interaction with an Octave Session
        """
        if not timeout == -1:
            self.set_timeout(timeout)

        if not self.proc:
            raise Oct2PyError('Session Closed, try a restart()')

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

        output = "disp(char(2));"
        output += """if execstr("%s;","errcatch") <>0 then;""" % expr
        output += "disp(lasterror()); disp(char(24));"
        output += "else; disp(char(3)); end;\n"

        if len(cmds) == 5:
            main_line = cmds[2].strip()
        else:
            main_line = '\n'.join(cmds)

        if 'keyboard' in output:
            self.write('keyboard\n')
            self.interact()
            return

        self.write(output)
        self.expect(chr(2))

        resp = []
        while 1:
            line = self.readline()

            if chr(3) in line:
                break

            elif chr(24) in line:
                msg = ('Oct2Py tried to run:\n"""\n{0}\n"""\n'
                       'Octave returned:\n{1}'
                       .format(main_line, '\n'.join(resp)))
                raise Oct2PyError(msg)

            elif line.endswith('> '):
                self.interact(line)

            if verbose and logger:
                logger.info(line)

            elif log and logger:
                logger.debug(line)

            if resp or line:
                resp.append(line)

        return '\n'.join(resp)

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
                raise Oct2PyError('Session Timed Out, closing')

    def write(self, message):
        """Write a message to the process using utf-8 encoding"""
        os.write(self.wfid, message.encode('utf-8'))

    def interact(self, prompt='debug> '):
        """Manage an Octave Debug Prompt interaction"""
        msg = 'Entering Octave Debug Prompt...\n%s' % prompt
        self.stdout.write(msg)
        while 1:
            inp_func = input if not PY2 else raw_input
            try:
                inp = inp_func() + '\n'
            except EOFError:
                return
            if inp in ['exit\n', 'quit\n', 'dbcont\n', 'dbquit\n',
                       'exit()\n', 'quit()\n']:
                inp = 'return\n'
            self.write('disp(char(3));' + inp)
            if inp == 'return\n':
                self.write('return\n')
                self.write('clear _\n')
                return
            self.expect('\x03')
            self.stdout.write(self.expect(prompt))

    def close(self):
        """Cleanly close an Octave session
        """
        try:
            self.proc.terminate()
        except (OSError, AttributeError):  # pragma: no cover
            pass
        self.proc = None


def _test():  # pragma: no cover
    """Run the doctests for this module.
    """
    print('Starting doctest')
    doctest.testmod()
    print('Completed doctest')


if __name__ == "__main__":  # pragma: no cover
    _test()
