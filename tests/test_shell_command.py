
from collections import OrderedDict
from unittest import TestCase, main

from lancet import ShellCommand


class TestShellCommand(TestCase):

    def test_kwargs_are_passed_correctly_ordered(self):
        """
        Test ShellCommand works correctly with an OrderedDict
        """
        # Given
        sc = ShellCommand('test.py')

        # When
        spec = OrderedDict([('arg1',0), ('arg2',1)])

        # Then
        cmd_line = sc(spec)
        expected = ['test.py', '--arg1', '0', '--arg2', '1']
        self.assertEqual(cmd_line, expected)

    def test_kwargs_are_passed_correctly_unordered(self):
        """
        Test ShellCommand works correctly with a regular dictionary
        """
        # Given
        sc = ShellCommand('test.py')

        # When
        spec = dict(arg1=0, arg2=1)

        # Then
        cmd_line = sc(spec)
        expected = ['test.py', '--arg1', '0', '--arg2', '1']
        # Always the first element
        self.assertEqual(cmd_line[0], 'test.py')
        # We cannot know if --arg1 or --arg2 will be presented first
        self.assertEqual(cmd_line.index('--arg1') < cmd_line.index('0'), True)
        self.assertEqual(cmd_line.index('--arg2') < cmd_line.index('1'), True)


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

    def test_expansions_are_correctly_called_for_long_filenames(self):
        # Given
        lf = ShellCommand.LongFilename('')
        sc = ShellCommand('test.py', expansions={'output': lf})

        # When
        spec = OrderedDict([('arg', 0)])
        info = {
            'root_directory': '/tmp',
            'batch_name': 'test',
            'varying_keys': ['arg']
        }

        # Then.
        result = sc(spec, info=info)

        fname = lf(spec, info, None)
        expected = ['test.py', '--output', fname, '--arg', '0']
        self.assertEqual(result, expected)


if __name__ == '__main__':
    main()
