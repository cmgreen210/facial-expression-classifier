import os
import pep8
import nose.tools as no
import unittest


class TestStyle(unittest.TestCase):

    def test_pep(self):
        """Test for PEP8 conformance"""

        root_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.join(root_dir, '..')

        file_list = []
        for root, _, files in os.walk(root_dir):
            for f in files:
                _, ext = os.path.splitext(f)
                if ext == '.py':
                    file_list.append(os.path.abspath(os.path.join(root, f)))

        pep8style = pep8.StyleGuide(quiet=False)

        result = pep8style.check_files(file_list)

        no.assert_equal(result.total_errors, 0,
                        result.print_statistics())
