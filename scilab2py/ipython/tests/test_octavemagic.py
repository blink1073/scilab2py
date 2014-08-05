"""Tests for Octave magics extension."""
import codecs
import unittest
import sys
from IPython.testing.globalipapp import get_ipython

try:
    import numpy.testing as npt
    from oct2py.ipython import octavemagic
except Exception:  # pragma: no cover
    __test__ = False


class OctaveMagicTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        '''Set up an IPython session just once.
        It'd be safer to set it up for each test, but for now,
        I'm mimicking the IPython team's logic.
        '''
        if not sys.stdin.encoding:
            # needed for py.test
            sys.stdin = codecs.getreader('utf-8')(sys.stdin)
        cls.ip = get_ipython()
        # This is just to get a minimally modified version of the changes
        # working
        cls.ip.magic('load_ext oct2py.ipython')
        cls.ip.ex('import numpy as np')
        cls.svgs_generated = 0

    def test_octave_inline(self):
        result = self.ip.run_line_magic('octave', '[1, 2, 3] + 1;')
        npt.assert_array_equal(result, [[2, 3, 4]])

    def test_octave_roundtrip(self):
        ip = self.ip
        ip.ex('x = np.arange(3); y = 4.5')
        ip.run_line_magic('octave_push', 'x y')
        ip.run_line_magic('octave', 'x = x + 1; y = y + 1;')
        ip.run_line_magic('octave_pull', 'x y')

        npt.assert_array_equal(ip.user_ns['x'], [[1, 2, 3]])
        npt.assert_equal(ip.user_ns['y'], 5.5)

    def test_octave_cell_magic(self):
        ip = self.ip
        ip.ex('x = 3; y = [1, 2]')
        ip.run_cell_magic('octave', '-f png -s 400,400 -i x,y -o z',
                          'z = x + y;')
        npt.assert_array_equal(ip.user_ns['z'], [[4, 5]])

    def test_octave_plot(self):
        magic = self.ip.find_cell_magic('octave').__self__
        magic._publish_display_data = self.verify_publish_data
        self.ip.run_cell_magic('octave', '-f svg -s 400,500',
            'plot([1, 2, 3]); figure; plot([4, 5, 6]);')
        npt.assert_equal(self.svgs_generated, 2)

    def verify_publish_data(self, source, data):
        if 'image/svg+xml' in data:
            svg = data['image/svg+xml']
            assert 'height="500px"' in svg
            assert 'width="400px"' in svg

            self.svgs_generated += 1

    def test_octavemagic_localscope(self):
        ip = self.ip
        ip.push({'x': 0})
        ip.run_line_magic('octave', '-i x -o result result = x+1')
        result = ip.user_ns['result']
        npt.assert_equal(result, 1)

        ip.run_cell('''def octavemagic_addone(u):
        %octave -i u -o result result = u+1
        return result''')
        ip.run_cell('result = octavemagic_addone(1)')
        result = ip.user_ns['result']
        npt.assert_equal(result, 2)

        npt.assert_raises(
            KeyError,
            ip.run_line_magic,
            "octave",
            "-i var_not_defined 1+1")

    def test_octave_syntax_error(self):
        try:
            self.ip.run_cell_magic('octave', '', "a='1")
        except octavemagic.OctaveMagicError:
            self.ip.magic('reload_ext oct2py.ipython')

    def test_octave_error(self):
        npt.assert_raises(octavemagic.OctaveMagicError, self.ip.run_cell_magic,
                          'octave', '', 'a = ones2(1)')
