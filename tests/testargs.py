"""
Unit tests for Lancet Args objects
"""

import os, sys
import unittest

cwd = os.path.abspath(os.path.split(__file__)[0])
sys.path.insert(0, os.path.join(cwd, '..'))
import lancet

if sys.version_info[0] == 2:
  from io import BytesIO as IO
else:
  from io import StringIO as IO

from contextlib import contextmanager

@contextmanager
def capture(command, *args, **kwargs):
    out, sys.stdout = sys.stdout, IO()
    command(*args, **kwargs)
    sys.stdout.seek(0)
    yield sys.stdout.read()
    sys.stdout = out


class TestArgSpecs(unittest.TestCase):
    def setUp(self):
        self.specs1 = [{'a':3, 'b':5}, {'a':4, 'b':6}]

        self.specs2 = [{'a':3.0, 'b':5.0, 'c':1.0},
                       {'a':2.0, 'b':6.0, 'c':2.0},
                       {'a':1.0, 'b':5.0, 'c':3.0}]

        self.specs3 = [{'a':3, 'b':5, 'c':1, 'd':5},
                       {'a':2, 'b':6, 'c':2, 'd':5},
                       {'a':1, 'b':5, 'c':3, 'd':5}]

        self.specs4 = [{'a':3, 'b':5, 'c':1, 'd':5},
                       {'a':2, 'b':5, 'c':2, 'd':5},
                       {'a':1, 'b':5, 'c':3, 'd':5},
                       {'a':1, 'b':6, 'c':4, 'd':5}]

        self.specs5 = [{'a':3.333333, 'b':5.555, 'c':1.1},
                       {'a':2.222222, 'b':6.666, 'c':2.2},
                       {'a':1.111111, 'b':5.555, 'c':3.3}]


class TestArgs(TestArgSpecs):

    def setUp(self):
        super(TestArgs, self).setUp()

    def test_args_kws(self):
        arg = lancet.Args(a=3)
        self.assertEqual(arg.specs, [{'a':3}])

    def test_args_specs(self):
        arg = lancet.Args(self.specs1)
        self.assertEqual(arg.specs, self.specs1)

    def test_args_iter(self):
        arg = lancet.Args(self.specs1)
        self.assertEqual([el for el in arg], [self.specs1])

    def test_args_next(self):
        arg = iter(lancet.Args(self.specs1))
        self.assertEqual(next(arg), self.specs1)

    def test_args_lexsort1(self):
        arg = lancet.Args(self.specs2)
        arg2 = arg.lexsort('+a', '+b', '+c')
        self.assertEquals(arg2.specs,
                          [{'a': 1, 'b': 5, 'c': 3},
                           {'a': 2, 'b': 6, 'c': 2},
                           {'a': 3, 'b': 5, 'c': 1}])

    def test_args_lexsort2(self):
        arg = lancet.Args(self.specs2)
        arg2 = arg.lexsort('-a', '-b', '-c')
        self.assertEquals(arg2.specs,
                          [{'a': 3, 'b': 5, 'c': 1},
                           {'a': 2, 'b': 6, 'c': 2},
                           {'a': 1, 'b': 5, 'c': 3}])

    def test_args_lexsort3(self):
        arg = lancet.Args(self.specs2)
        arg2 = arg.lexsort('+b', '-a')
        self.assertEquals(arg2.specs,
                          [{'a': 3, 'b': 5, 'c': 1},
                           {'a': 1, 'b': 5, 'c': 3},
                           {'a': 2, 'b': 6, 'c': 2}])

    def test_args_lexsort4(self):
        arg = lancet.Args(self.specs2)
        arg2 = arg.lexsort('+b', '-c')
        self.assertEquals(arg2.specs,
                          [{'a': 1, 'b': 5, 'c': 3},
                           {'a': 3, 'b': 5, 'c': 1},
                           {'a': 2, 'b': 6, 'c': 2}])

    def test_args_constant_keys(self):
        arg = lancet.Args(self.specs3)
        self.assertEqual(arg.constant_keys, ['d'])

    def test_args_constant_items(self):
        arg = lancet.Args(self.specs3)
        self.assertEqual(arg.constant_items, [('d', 5)])

    def test_args_varying_keys(self):
        arg = lancet.Args(self.specs4)
        self.assertEqual(arg.varying_keys, ['b', 'a', 'c'])

    def test_args_len1(self):
        arg = lancet.Args(self.specs1)
        self.assertEqual(len(arg), 2)

    def test_args_len2(self):
        arg = lancet.Args(self.specs3)
        self.assertEqual(len(arg), 3)

    def test_args_len3(self):
        arg = lancet.Args(self.specs4)
        self.assertEqual(len(arg), 4)

    def test_args_str1(self):
        arg = lancet.Args(a=4, b=6)
        self.assertEquals(str(arg), 'Args(\n   a=4,\n   b=6\n)')

    def test_args_str2(self):
        """
        As OrderedDicts not used internally, the dictionaries shown in
        the full specification list can change between calls.
        """
        arg = lancet.Args(self.specs1)
        self.assertEquals(str(arg).startswith('Args(\n   specs=[{'), True)
        self.assertEquals(str(arg).endswith('}]\n)'), True)

    def test_args_repr1(self):
        arg = lancet.Args(a=4, b=6)
        self.assertEquals(repr(arg),
                          'Args(fp_precision=4,a=4,b=6)')

    def test_args_repr2(self):
        """
        As OrderedDicts not used internally, the dictionaries shown in
        the full specification list can change between calls.
        """
        arg = lancet.Args(self.specs1)
        self.assertEquals(repr(arg).startswith('Args(specs=[{'), True)
        self.assertEquals(repr(arg).endswith('}],fp_precision=4)'), True)

    def test_args_show(self):
        arg = lancet.Args(self.specs1)
        with capture(arg.show) as out:
            self.assertEquals(out, '0: a=3, b=5\n1: a=4, b=6\n')

    def test_args_show_fp_precision0(self):
        arg = lancet.Args(self.specs2, fp_precision=0)
        with capture(arg.show) as out:
            self.assertEquals(out, '0: a=3, b=5, c=1\n1: a=2, b=6, c=2\n2: a=1, b=5, c=3\n')


    def test_args_show_fp_precision2(self):
        arg = lancet.Args(self.specs5, fp_precision=2)
        expected = '0: a=3.33, b=5.55, c=1.1\n1: a=2.22, b=6.67, c=2.2\n2: a=1.11, b=5.55, c=3.3\n'
        with capture(arg.show) as out:
            self.assertEquals(out, expected)

    def test_args_show_fp_precision6(self):
        arg = lancet.Args(self.specs5, fp_precision=6)
        expected = ('0: a=3.333333, b=5.555, c=1.1\n1: a=2.222222, b=6.666,'
                    ' c=2.2\n2: a=1.111111, b=5.555, c=3.3\n')
        with capture(arg.show) as out:
            self.assertEquals(out, expected)

    def test_args_contains(self):
      arg = lancet.Args(self.specs5)
      self.assertEqual('a' in arg, True)

    def test_args_doesnt_contains(self):
      arg = lancet.Args(self.specs5)
      self.assertEqual('z' in arg, False)

    def test_args_summary1(self):
      arg = lancet.Args(self.specs1)
      with capture(arg.summary) as out:
        self.assertEquals(out, "Items: 2\nVarying Keys: 'a', 'b'\n")

    def test_args_summary2(self):
      arg = lancet.Args(self.specs2)
      with capture(arg.summary) as out:
        self.assertEquals(out, "Items: 3\nVarying Keys: 'a', 'b', 'c'\n")

    def test_args_summary3(self):
      arg = lancet.Args(self.specs3)
      with capture(arg.summary) as out:
        self.assertEquals(out, "Items: 3\nVarying Keys: 'a', 'b', 'c'\nConstant Items: d=5\n")

    def test_args_summary4(self):
      arg = lancet.Args(self.specs4)
      with capture(arg.summary) as out:
        self.assertEquals(out, "Items: 4\nVarying Keys: 'b', 'a', 'c'\nConstant Items: d=5\n")

    def test_args_summary5(self):
      arg = lancet.Args(self.specs5)
      with capture(arg.summary) as out:
        self.assertEquals(out, "Items: 3\nVarying Keys: 'a', 'b', 'c'\n")

    def test_args_copy(self):
      arg = lancet.Args(self.specs1)
      self.assertEqual(arg.specs, arg.copy().specs)


class TestArgsCompose(TestArgSpecs):

    def setUp(self):
        super(TestArgsCompose, self).setUp()

    def test_concatenate(self):
      arg1 = lancet.Args(self.specs1)
      arg2 = lancet.Args(self.specs2)
      self.assertEqual((arg1 + arg2).specs, self.specs1 + self.specs2)


if __name__ == "__main__":
    import sys
    import nose
    nose.runmodule(argv=[sys.argv[0], "--logging-level", "ERROR"])

