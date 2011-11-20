"""
.. module:: _h5read
   :synopsis: Read Python values from an HDF file made by Octave.
              Strives to preserve both value and type in transit.

.. moduleauthor:: Steven Silvester <steven.silvester@ieee.org>

"""
try:
    import h5py
except:
    print 'Please install h5py from'
    print '"http://code.google.com/p/h5py/downloads/list"'
    raise
import numpy as np
from _utils import Struct, _create_hdf, _register_del


class H5Read(object):
    """Read Python values from an HDF file made by Octave.

    Strives to preserve both value and type in transit.

    """
    def __init__(self):
        """Initialize our output file and register it for deletion
        """
        self.out_file = _create_hdf('save')
        _register_del(self.out_file)

    def setup(self, nout, names=None):
        """
        Generate the argout list and the Octave save command.

        Parameters
        ----------
        nout : int
            Number of output arguments required.
        names : array-like, optional
            Variable names to use.

        Returns
        -------
        out : tuple (list, str)
            List of variable names, Octave "save" command line

        """
        argout_list = []
        for i in range(nout):
            if names:
                argout_list.append(names.pop(0))
            else:
                argout_list.append("%s__" % chr(i + 97))
        save_line = 'save "-hdf5" "%s" "%s"' % (self.out_file,
                                                '" "'.join(argout_list))
        return argout_list, save_line

    def extract_file(self, argout_list):
        """
        Extract the variables in argout_list from the HDF file

        Parameters
        ----------
        argout_list : array-like
            List of variables to extract from the file

        Returns
        -------
        out : object or tuple
            Variable or tuple of variables extracted.

        """
        fid = h5py.File(self.out_file)
        outputs = []
        for arg in argout_list:
            try:
                val = self._getval(fid[arg])
            except AttributeError:
                val = self._getvals(fid[arg]['value'])
            outputs.append(val)
        fid.close()
        if len(outputs) > 1:
            return tuple(outputs)
        else:
            return outputs[0]

    @staticmethod
    def _getval(group):
        """
        Handle variable types that do not translate directly.

        Parameters
        ----------
        group : h5py Group object
            The location from which to extract the value from the file.

        Returns
        =======
        out : object
            Python object extracted from file.

        """
        type_ = group['type'].value
        val = group['value'].value
        # strings come in as byte arrays
        if type_ == 'sq_string' or type_ == 'string':
            try:
                val = [chr(char) for char in val]
                val = ''.join(val)
            except TypeError:
                temp = [chr(item) for item in val.ravel()]
                temp = np.array(temp).reshape(val.shape)
                val = []
                for row in range(temp.shape[1]):
                    val.append(''.join(temp[:, row]))
        # complex scalars come in as tuples
        elif type_ == 'complex scalar':
            val = val[0] + val[1] * 1j
        # complex matrices come in as ndarrays with real and imag parts
        elif type_ == 'complex matrix':
            temp = [x + y * 1j for x, y in val.ravel()]
            val = np.array(temp).reshape(val.shape)
        # Matlab reads the data in Fortran order, not 'C' order
        if isinstance(val, np.ndarray):
            val = val.T
        return val

    def _getvals(self, group):
        """
        Extract a nested struct / cell array from the HDF file.

        Structs become dictionaries, cell arrays become lists.

        Parameters
        ==========
        group : h5py group object
            Location from which to extract the values from the file.

        Returns
        =======
        out : object or dict
            Object(s) extracted from file.

        """
        data = Struct()
        for key in group.keys():
            if key == 'dims':
                data['dims'] = group[key].value
            elif isinstance(group[key]['value'], h5py.Group):
                if key.startswith('_'):
                    data[int(key[1:])] = self._getvals(group[key]['value'])
                else:
                    data[key] = self._getvals(group[key]['value'])
            else:
                val = self._getval(group[key])
                if key.startswith('_'):
                    key = int(key[1:])
                data[key] = val
        # handle cell arrays
        if 'dims' in data:
            data = self._extract_cell_array(data)
        return data

    @staticmethod
    def _extract_cell_array(data):
        """
        Extract a nested cell array from a dictionary.

        Parameters
        ==========
        data : dict
            Cell array data in a dictionary.

        Returns
        =======
        out : scalar, list, or list of lists
            Return the array contents, matching the shape of the
            cell array.

        """
        dims = data['dims']
        # only worry about 1-d and 2-d
        if len(dims) == 2:
            # singleton
            if dims[0] == 1 and dims[1] == 1:
                data = data[0]
            # array
            elif dims[0] == 1 or dims[1] == 1:
                del data['dims']
                data = [data[key] for key in sorted(data.keys())]
            # matrix
            else:
                temp = []
                for row in range(dims[0]):
                    start = row * dims[1]
                    stop = (row + 1) * dims[1]
                    temp.append([data[key] for key in range(start, stop)])
                data = temp
        return data
