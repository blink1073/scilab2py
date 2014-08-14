"""
.. module:: matread
   :synopsis: Read Python values from a MAT file made by Scilab.
              Strives to preserve both value and type in transit.

.. moduleauthor:: Steven Silvester <steven.silvester@ieee.org>

"""
import os
import numpy as np
from scipy.io import loadmat
import scipy
from .utils import Struct, create_file


class MatRead(object):
    """Read Python values from a MAT file made by Scilab.

    Strives to preserve both value and type in transit.

    """
    def __init__(self, temp_dir=None):
        """Initialize our output file
        """
        self.temp_dir = temp_dir
        self.out_file = create_file(self.temp_dir)

    def setup(self, nout, names=None):
        """
        Generate the argout list and the Scilab save command.

        Parameters
        ----------
        nout : int
            Number of output arguments required.
        names : array-like, optional
            Variable names to use.

        Returns
        -------
        out : tuple (list, str)
            List of variable names, Scilab "savematfile" command line

        """
        argout_list = []
        for i in range(nout):
            if names:
                argout_list.append(names.pop(0))
            else:
                argout_list.append("%s__" % chr(i + 97))
        save_line = 'savematfile -v6 {0} {1}'
        save_line = save_line.format(self.out_file,
                                     ' '.join(argout_list))
        return argout_list, save_line

    def remove_file(self):
        try:
            os.remove(self.out_file)
        except (OSError, AttributeError):  # pragma: no cover
            pass

    def extract_file(self, variables=None):
        """
        Extract the variables in argout_list from the M file

        Parameters
        ----------
        argout_list : array-like
            List of variables to extract from the file

        Returns
        -------
        out : object or tuple
            Variable or tuple of variables extracted.

        """
        data = loadmat(self.out_file)
        for key in list(data.keys()):
            if key.startswith('_') and not key == '_':
                del data[key]
            else:
                data[key] = get_data(data[key])
        if len(data) == 1:
            return list(data.values())[0]
        elif data:
            return data


def get_data(val):
    '''Extract the data from the incoming value
    '''
    if not isinstance(val, np.ndarray):
        return val
    # check for objects
    if "'|O" in str(val.dtype) or "O'" in str(val.dtype):
        data = Struct()
        for key in val.dtype.fields.keys():
            data[key] = get_data(val[key][0])
        return data
    # handle cell arrays
    if val.dtype == np.object:
        if val.size == 1:
            val = val[0]
            if "'|O" in str(val.dtype) or "O'" in str(val.dtype):
                val = get_data(val)
            if isinstance(val, Struct):
                return val
            if val.size == 1:
                val = val.flatten()
    if val.dtype == np.object:
        if len(val.shape) > 2:
            val = val.T
            val = np.array([get_data(val[i].T)
                            for i in range(val.shape[0])])
        if len(val.shape) > 1:
            if len(val.shape) == 2:
                val = val.T
            try:
                return val.astype(val[0][0].dtype)
            except ValueError:
                # dig into the cell type
                for row in range(val.shape[0]):
                    for i in range(val[row].size):
                        if not np.isscalar(val[row][i]):
                            if val[row][i].size > 1:
                                val[row][i] = val[row][i].squeeze()
                            else:
                                val[row][i] = val[row][i][0]
            except AttributeError:
                return val

        else:
            val = np.array([get_data(val[i])
                            for i in range(val.size)])
        if len(val.shape) == 1 or val.shape[0] == 1 or val.shape[1] == 1:
            val = val.flatten()
        val = val.tolist()
        if len(val) == 1 and isinstance(val[0],
                                        scipy.sparse.csc.csc_matrix):
            val = val[0]
    elif val.size == 1:
        if hasattr(val, 'flatten'):
            val = val.flatten()[0]
    return val
