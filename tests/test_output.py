import os
import shutil
import sys
import tempfile
from unittest import TestCase, main

import lancet

class TestOutput(TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.root, 'test')

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_outputs_captured_correctly(self):
        # Given
        nargs = 2
        args = lancet.List('c', ["print(%d)"%i for i in range(nargs)])
        cmd = lancet.ShellCommand(executable=sys.executable)
        lancet.Launcher('test', args, cmd, output_directory=self.output_dir)()

        # When
        o = lancet.Output(self.output_dir)
        latest = o[-1]

        # Then
        self.assertEqual(len(o), 1)
        expected = (
            'timestamp', 'path', 'tids', 'specs', 'stdout', 'stderr',
            'log', 'info'
        )
        self.assertEqual(latest._fields, expected)
        self.assertEqual(latest.tids, list(range(nargs)))
        self.assertEqual(len(latest.stdout), nargs)
        self.assertEqual(len(latest.stderr), nargs)
        expected = [{'c':'print(%d)'%i} for i in range(nargs)]
        self.assertEqual(latest.specs, expected)

        # Gather the stdout data.
        result = [int(open(x).read().strip()) for x in latest.stdout]
        expected = list(range(nargs))
        self.assertEqual(result, expected)

        # We should be able to iterate over the output.
        paths = [x.path for x in o]
        expected = [latest.path]
        self.assertEqual(paths, expected)

    def test_output_supports_expansions(self):
        # Given
        nargs = 2
        args = lancet.List('c', ["print(%d)"%i for i in range(nargs)])
        cmd = lancet.ShellCommand(executable=sys.executable)
        lancet.Launcher('test', args, cmd, output_directory=self.output_dir)()

        # This is useful if one needs to generate a separate file for each
        # invocation.  Here a filename is generated for each case run.
        expansions = {'filename': lancet.ShellCommand.LongFilename('.png')}

        # When
        o = lancet.Output(self.output_dir, expansions=expansions)
        latest = o[-1]

        # Then.
        self.assertEqual(len(o), 1)
        expected = (
            'timestamp', 'path', 'tids', 'specs', 'stdout', 'stderr',
            'log', 'info', 'filename'
        )
        self.assertEqual(latest._fields, expected)
        self.assertEqual(len(latest.filename), 2)
        self.assertTrue(all([x.endswith('.png') for x in latest.filename]))


if __name__ == '__main__':
    main()
