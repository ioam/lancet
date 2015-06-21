
from collections import OrderedDict
from glob import glob
import json
import os
from os.path import isdir, join, splitext

import lancet.core as core


class Output(object):
    """A convenience class to help collect the generated outputs from an
    invocation of a `Launcher`.

    Given an output directory, the object reads the log file, info file and
    provides easy access to the output streams.

    The important attributes are:

     - self.info: The info dict for the launch.
     - self.log: The specs as an ordered dict {tid:spec}
     - self.output_dir: the output directory.
     - self.specs: list of specifications in same order as tids.
     - self.stderr: list of stderr filenames.
     - self.stdout: list of stdout filenames.
     - self.tids: list of task ids.

    The object also allows users to perform expansions like the ShellCommand,
    see `do_expansion`.

    """

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self._read_log()
        self._read_info()
        self._get_streams()

    def _read_log(self):
        log_path = glob(join(self.output_dir, '*.log'))[0]
        self.log = core.Log.extract_log(log_path, OrderedDict)
        self.tids = list(self.log.keys())
        self.specs = list(self.log.values())

    def _read_info(self):
        info_file = glob(join(self.output_dir, '*.info'))[0]
        self.info = json.load(open(info_file))

    def _get_streams(self):

        def _get_paths(pattern):
            streams = join(self.info['root_directory'], 'streams')
            files = glob(join(streams, pattern))
            files = sorted(files, key=lambda x: splitext(x)[1])
            return files

        batch_name = self.info['batch_name']
        self.stdout = _get_paths('%s_*.o.*'%batch_name)
        self.stderr = _get_paths('%s_*.e.*'%batch_name)

    def do_expansion(self, func):
        """Perform expansions (analogous to a ShellCommand) given a callable.

        The callable will be passed positional arguments (spec, info, tid) for
        each invocation of the command and returns the result as an ordered
        dictionary with keys as the tids and results as the values.

        This is especially useful when one wishes to collect output files
        using `ShellCommand.LongFilename`.
        """
        return [func(self.specs[tid], self.info, tid) for tid in self.tids]


class CollectOutputs(object):

    """Collects a dictionary of outputs from various invocations of the
    launcher from a given directory.

    Attributes:

        - self.outputs: an ordered dictionary of {directory:Output}

    The `get_latest` method returns the latest output.

    """

    def __init__(self, output_dir='.'):
        """The output_dir is the directory containing the output of the
        launcher.
        """

        outputs = OrderedDict()
        for path in sorted(os.listdir(output_dir)):
            full_path = join(output_dir, path)
            if isdir(full_path):
                outputs[path] = Output(full_path)
        self.outputs = outputs

    def get_latest(self):
        """Return the latest `Output` object available in the given directory.
        """
        return self.outputs.values()[-1]
