
from collections import OrderedDict
from unittest import TestCase, main

from lancet import ShellCommand


class TestShellCommand(TestCase):
    def test_kwargs_are_passed_correctly(self):
        # Given
        sc = ShellCommand('test.py')

        # When
        spec = OrderedDict(arg1=0, arg2=1)

        # Then
        cmd_line = sc(spec)
        expected = ['test.py', '--arg1', '0', '--arg2', '1']
        self.assertEqual(cmd_line, expected)

    def test_kwargs_with_true_value_is_passed_correctly(self):
        # Given
        sc = ShellCommand('test.py')

        # When
        spec = OrderedDict(arg=True)

        # Then
        cmd_line = sc(spec)
        expected = ['test.py', '--arg']
        self.assertEqual(cmd_line, expected)

    def test_kwargs_with_false_value_is_not_passed(self):
        # Given
        sc = ShellCommand('test.py')

        # When
        spec = OrderedDict(arg=False)

        # Then
        cmd_line = sc(spec)
        expected = ['test.py']
        self.assertEqual(cmd_line, expected)

if __name__ == '__main__':
    main()
