
from collections import OrderedDict, namedtuple
from glob import glob
import json
import os
from os.path import isdir, join, splitext

import param

import lancet.core as core


class Output(param.Parameterized):
    """A convenience class to help collect the generated outputs from an
    invocation of a `Launcher` (which is called a "launch").

    Given an output directory, the object reads the log file, info file and
    provides easy access to the output streams for each of the launches.

    The Launch information for each launch is stored in a namedtuple called
    `LaunchInfo` which has the following fields::

      'timestamp', 'path', 'tids', 'specs', 'stdout', 'stderr', 'log', 'info'

    If there are any expansions specified (see the ShellCommand's expansions
    parameter), those are also added to the named tuple.

    Here is an example of using the class::

        >>> output = Output('output')
        >>> len(output)
        2
        >>> output[-1]._fields # the fields of the namedtuple.
        ('timestamp', 'path', 'tids', 'specs', 'stdout', 'stderr', 'log', 'info')

        >>> output[-1].path # the path of the last run.
        u'/tmp/output/2015-06-21_0325-prime_quintuplet'

        >>> len(output[-1].specs) # spec the arguments for each case.
        16
        >>> output[-1].specs[-1]
        {'integer': 115}
        >>> len(output[1].stdout) # the path to the stdout for each case.
        16
        >>> open(output[1].stdout[-1]).read()
        '109: 109\n'

    One can iterate over the LaunchInfo for the launches like so::

        >>> for li in output:
        ...     print(li.path)
        ...
        /tmp/output/2015-06-21_0315-prime_quintuplet
        /tmp/output/2015-06-21_0325-prime_quintuplet
        >>>

    """

    output_dir = param.String(default='.', doc='''
         The output directory - the directory that will contain all
         the root directories for the individual launches.''')

    expansions = param.Dict(default={}, constant=True, doc='''
        Perform expansions (analogous to a ShellCommand) given a callable.
        Allows extension of the specification that supports callables that
        expand to valid argument values.  The callable must have the signature
        (spec, info, tid). A typical usage for a function value is to build a
        valid output filename given the context.

        See the `ShellCommand.RootDirectory`, `ShellCommand.LongFilename` and
        `ShellCommand.Expand`.''')

    ##### object Protocol ####################################################

    def __init__(self, output_dir, **params):
        super(Output,self).__init__(output_dir=output_dir, **params)
        self.launches = []
        self.expansion_keys = sorted(self.expansions.keys())
        self.LaunchInfo = namedtuple(
            'LaunchInfo', [
                'timestamp', 'path', 'tids', 'specs', 'stdout', 'stderr',
                'log', 'info'
            ] + self.expansion_keys
        )
        self.update()

    def __getitem__(self, launch):
        return self.launches[launch]

    def __iter__(self):
        return iter(self.launches)

    def __len__(self):
        return len(self.launches)

    ##### Private Protocol ####################################################

    def _get_launch_info(self, launch_dir):
        log_path = glob(join(launch_dir, '*.log'))[0]
        log = core.Log.extract_log(log_path, OrderedDict)
        tids = list(log.keys())
        specs = list(log.values())

        info_file = glob(join(launch_dir, '*.info'))[0]
        info = json.load(open(info_file))
        stdout, stderr = self._get_streams(info)

        info_dict = dict(
            timestamp=info['timestamp'],
            path=info['root_directory'], tids=tids, specs=specs,
            stdout=stdout, stderr=stderr, log=log, info=info
        )

        # Do all the expansions.
        for name, func in self.expansions.items():
            info_dict[name] = [func(specs[tid], info, tid) for tid in tids]

        launch_info = self.LaunchInfo(**info_dict)
        return launch_info

    def _get_streams(self, info):

        def _get_paths(pattern):
            streams = join(info['root_directory'], 'streams')
            files = glob(join(streams, pattern))
            files = sorted(files, key=lambda x: splitext(x)[1])
            return files

        batch_name = info['batch_name']
        stdout = _get_paths('%s_*.o.*'%batch_name)
        stderr = _get_paths('%s_*.e.*'%batch_name)
        return stdout, stderr

    ##### Public  Protocol ####################################################

    def update(self):
        """Update the launch information -- use if additional launches were
        made.
        """
        launches = []
        for path in sorted(os.listdir(self.output_dir)):
            full_path = join(self.output_dir, path)
            if isdir(full_path):
                launches.append(self._get_launch_info(full_path))
        launches.sort()
        self.launches = launches
