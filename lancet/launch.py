#
# Lancet launchers and launch helpers
#

import os, sys, platform, time, pipes, subprocess, types
import json, pickle

import param

import lancet.core as core
from lancet import __version__ as lancet_version
from lancet.dynamic import DynamicArgs

# For Python 2 and 3 compatibility
try:
    input = raw_input
except NameError:
    pass

#===================#
# Commands Template #
#===================#

class Command(core.PrettyPrinted, param.Parameterized):
    """
    A command is a way of converting the dictionaries returned by
    argument specifiers into a particular command. When called with an
    argument specifier, a command template returns a list of strings
    corresponding to a subprocess Popen argument list.

    __call__(self, spec, tid=None, info={}):

    All Commands must be callable. The tid argument is the task id and
    info is a dictionary of run-time information supplied by the
    launcher. See the _setup_launch method of Launcher to see details
    about the launch information supplied.
    """

    executable = param.String(default='python', constant=True, doc='''
        The executable that is to be run by this Command. Unless the
        executable is a standard command expected on the system path,
        this should be an absolute path. By default this invokes
        python or the python environment used to invoke the Command
        (Topographica for instance).''')

    do_format = param.Boolean(default=True, doc= '''
        Set to True to receive input arguments as formatted strings,
        False for the raw unformatted objects.''')

    def __init__(self, executable=None, **params):
        if executable is None:
            executable = sys.executable
        self._pprint_args = ([],[],None,{})
        super(Command,self).__init__(executable=executable, **params)
        self.pprint_args([],[])

    def __call__(self, spec, tid=None, info={}):
        """
        Formats a single argument specification supplied as a
        dictionary of argument name/value pairs. The info dictionary
        contains launch information as defined in the _setup_launch
        method of Launcher.
        """
        raise NotImplementedError

    def _formatter(self, spec):
        if self.do_format: return core.Arguments.spec_formatter(spec)
        else             : return spec

    def show(self, args, file_handle=None, **kwargs):
        "Write to file_handle if supplied, othewise print output"
        full_string = ''
        info = {'root_directory':     '<root_directory>',
                'batch_name':         '<batch_name>',
                'batch_tag':          '<batch_tag>',
                'batch_description':  '<batch_description>',
                'launcher':        '<launcher>',
                'timestamp_format':   '<timestamp_format>',
                'timestamp':          tuple(time.localtime()),
                'varying_keys':       args.varying_keys,
                'constant_keys':      args.constant_keys,
                'constant_items':     args.constant_items}

        quoted_cmds = [ subprocess.list2cmdline(
                [el for el in self(self._formatter(s),'<tid>',info)])
                        for s in args.specs]

        cmd_lines = ['%d: %s\n' % (i, qcmds) for (i,qcmds)
                     in enumerate(quoted_cmds)]
        full_string += ''.join(cmd_lines)
        if file_handle:
            file_handle.write(full_string)
            file_handle.flush()
        else:
            print(full_string)

    def verify(self, args):
        """
        Optional, final check that ensures valid arguments have been
        passed before launch. Allows the constant and varying_keys to
        be be checked and can inspect the specs attribute if an
        instance of Args. If invalid, raise an Exception with the
        appropriate error message, otherwise return None.
        """
        return

    def finalize(self, info):
        """
        Optional method that allows a Command to save state before
        launch. The info argument is supplied by the Launcher.
        """
        return

    def summary(self):
        """
        A succinct summary of the Command configuration.  Unlike the
        repr, a summary does not have to be complete but must supply
        key information relevant to the user. Must begin by stating
        the executable.
        """
        raise NotImplementedError


class ShellCommand(Command):
    """
    A generic Command that can be used to invoke shell commands on
    most operating systems where Python can be run. By default,
    follows the GNU coding convention for commandline arguments.
    """

    expansions = param.Dict(default={}, constant=True, doc='''
        Allows extension of the specification that supports functions
        that expand to valid argument values.  If a function is used,
        it must have the signature (spec, info, tid). A typical usage
        for a function value is to build a valid output filename given
        the context.

        Three such subclasses are provided:
        'RootDirectory', 'LongFilename' and 'Expand'.''')

    posargs = param.List(default=[], constant=True, doc='''
       The list of positional argument keys. Positional arguments are
       always supplied at the end of a command in the order given.''')

    short_prefix = param.String(default='-',  constant=True, doc='''
       Although the single dash is a GNU coding convention, the
       argument prefix may depend on the applications and/or platform.''')

    long_prefix = param.String(default='--',  constant=True, doc='''
       Although the double dash is a GNU coding convention, some
       applications use single dashes for long options.''')

    def __init__(self, executable, **params):
        super(ShellCommand,self).__init__(executable = executable,
                                          do_format=False,
                                          **params)
        self.pprint_args(['executable','posargs'],['long_prefix'])

    def __call__(self, spec, tid=None, info={}):
        # Function expansions are called here.
        expanded = type(spec)()
        for (k,v) in self.expansions.items():
            if callable(v):
                expanded[k] = v(spec, info, tid)
            else:
                expanded[k] = v

        expanded.update(spec.items())
        expanded = core.Arguments.spec_formatter(expanded)

        options = []
        for (k, v) in expanded.items():
            if k in self.posargs or spec.get(k) is False:
                continue
            options.append('%s%s' % (self.long_prefix if len(k) > 1 else self.short_prefix, k))
            if spec.get(k) is not True:
                options.append(v)

        posargs = [expanded[parg] if (parg in expanded) else parg(spec, info, tid)
                   for parg in self.posargs]
        return [self.executable] + options + posargs

    def summary(self):
        print("Command executable: %s" % self.executable)
        print("Long prefix: %r" % self.long_prefix)

    class RootDirectory(object):
        """
        Supplies the root_directory to a command.
        """
        def __call__(self, spec, info, tid):
            return  info['root_directory']

        def __repr__(self):
            return "ShellCommand.RootDirectory()"

    class LongFilename(object):
        """
        Generates a long filename based on the input arguments in the
        root directory with the given extension. Ignores constant
        items.
        """
        def __init__(self, extension, excluding=[]):
            self.extension = extension
            self.excluding = excluding

        def __call__(self, spec, info, tid):
            root_dir = info['root_directory']
            params = [('tid' , tid)] + [(k,v) for  (k,v) in spec.items()
                                        if k in info['varying_keys']
                                        and k not in self.excluding]
            basename = '_'.join('%s=%s' % (k,v) for (k,v) in sorted(params))
            return os.path.join(root_dir, '%s_%s%s' % (info['batch_name'],
                                                       basename,
                                                       self.extension))
        def __repr__(self):
            items = ([self.extension, self.excluding]
                     if self.excluding else [self.extension])
            return ("ShellCommand.LongFilename(%s)"
                    % ', '.join('%r' % el for el in items))

    class Expand(object):
        """
        Takes a new-style format string template and expands it out
        using the keys in the spec, info and tid.
        """

        def __init__(self, template):
            self.template = template

        def __call__(self,spec, info, tid):
            all_params = {'tid' : tid}
            all_params.update(spec)
            all_params.update(info)
            return self.template.format(**all_params)

        def __repr__(self):
            return "ShellCommand.Expand(%r)" % self.template


#===========#
# Launchers #
#===========#

class Launcher(core.PrettyPrinted, param.Parameterized):
    """
    A Launcher is constructed using a name, an argument specifier and
    a command template. It can then launch the corresponding tasks
    appropriately when invoked.

    This default Launcher uses subprocess to launch tasks. It is
    intended to illustrate the basic design and should be used as a
    base class for more complex Launchers. In particular all Launchers
    should retain the same behaviour of writing stdout/stderr to the
    streams directory, writing a log file and recording launch
    information.
    """

    batch_name = param.String(default=None, allow_None=True, constant=True,
       doc='''A unique identifier for the current batch''')

    args = param.ClassSelector(core.Arguments, constant=True, doc='''
       The specifier used to generate the varying parameters for the tasks.''')

    command = param.ClassSelector(Command, constant=True, doc='''
       The command template used to generate the commands for the current tasks.''')

    output_directory = param.String(default='.', doc='''
         The output directory - the directory that will contain all
         the root directories for the individual launches.''')

    subdir = param.List(default=[], doc='''
      A list of subdirectory names that allows custom organization
      within the output directory before the root directory.''')

    tag = param.String(default='', doc='''
       A very short, identifiable human-readable string that
       meaningfully describes the batch to be executed. Should not
       include spaces as it may be used in filenames.''')

    description = param.String(default='', doc='''
       A short description of the purpose of the current set of tasks.''')

    metadata = param.Dict(default={}, doc='''
       Metadata information to add to the info file.''')

    max_concurrency = param.Integer(default=2, allow_None=True, doc='''
       Concurrency limit to impose on the launch. As the current class
       uses subprocess locally, multiple processes are possible at
       once. Set to None for no limit (eg. for clusters)''')

    reduction_fn = param.Callable(default=None, doc='''
      A callable that will be invoked when the Launcher has completed
      all tasks. For example, this could inform the user of completion
      (eg. send an e-mail) among other possibilities.''')

    timestamp = param.NumericTuple(default=(0,)*9, doc='''
      Optional override of timestamp (default timestamp set on launch
      call) in Python struct_time 9-tuple format.  Useful when you
      need to have a known root_directory path (see root_directory
      documentation) before launch. For example, you should store
      state related to analysis (eg. pickles) in the same location as
      everything else.''')

    timestamp_format = param.String(default='%Y-%m-%d_%H%M', allow_None=True, doc='''
      The timestamp format for the root directories in python datetime
      format. If None, the timestamp is omitted from root directory
      name.''')


    def __init__(self, batch_name, args, command, **params):

        self._pprint_args = ([],[],None,{})
        if 'name' not in params: params['name'] = self.__class__.__name__
        super(Launcher,self).__init__(batch_name=batch_name,
                                      args=args,
                                      command = command,
                                      **params)
        self._spec_log = []
        if self.timestamp == (0,)*9:
            self.timestamp = tuple(time.localtime())
        self.pprint_args(['batch_name','args','command'],
                         ['description', 'tag', 'output_directory',
                          'subdir','metadata'])
        self.dynamic = isinstance(args, DynamicArgs)

    def get_root_directory(self, timestamp=None):
        """
        A helper method that supplies the root directory name given a
        timestamp.
        """
        if timestamp is None: timestamp = self.timestamp
        if self.timestamp_format is not None:
            root_name =  (time.strftime(self.timestamp_format, timestamp)
                          + '-' + self.batch_name)
        else:
            root_name = self.batch_name

        path = os.path.join(self.output_directory,
                                *(self.subdir+[root_name]))
        return os.path.abspath(path)

    def _append_log(self, specs):
        """
        The log contains the tids and corresponding specifications
        used during launch with the specifications in JSON format.
        """
        self._spec_log += specs # This should be removed
        log_path = os.path.join(self.root_directory, ("%s.log" % self.batch_name))
        core.Log.write_log(log_path, [spec for (_, spec) in specs], allow_append=True)

    def _record_info(self, setup_info=None):
        """
        All launchers should call this method to write the info file
        at the end of the launch. The .info file is saved given
        setup_info supplied by _setup_launch into the
        root_directory. When called without setup_info, the existing
        info file is updated with the end-time.
        """
        info_path = os.path.join(self.root_directory, ('%s.info' % self.batch_name))

        if setup_info is None:
            try:
                with open(info_path, 'r') as info_file:
                    setup_info = json.load(info_file)
            except:
                setup_info = {}

            setup_info.update({'end_time' : tuple(time.localtime())})
        else:
            setup_info.update({
                'end_time' : None,
                'metadata' : self.metadata
                })

        with open(info_path, 'w') as info_file:
            json.dump(setup_info, info_file, sort_keys=True, indent=4)

    def _setup_launch(self):
        """
        Method to be used by all launchers that prepares the root
        directory and generate basic launch information for command
        templates to use (including a registered timestamp).
        """
        self.root_directory = self.get_root_directory()
        if not os.path.isdir(self.root_directory):
            os.makedirs(self.root_directory)

        platform_dict = {}
        python_version = (platform.python_implementation()
                          + platform.python_version())
        platform_dict['platform']       = platform.platform()
        platform_dict['python_version'] = python_version
        platform_dict['lancet_version'] = str(lancet_version)

        return {'root_directory':    self.root_directory,
                'batch_name':        self.batch_name,
                'batch_tag':         self.tag,
                'batch_description': self.description,
                'launcher':          repr(self),
                'platform' :         platform_dict,
                'timestamp':         self.timestamp,
                'timestamp_format':  self.timestamp_format,
                'varying_keys':      self.args.varying_keys,
                'constant_keys':     self.args.constant_keys,
                'constant_items':    self.args.constant_items}


    def _setup_streams_path(self):
        streams_path = os.path.join(self.root_directory, "streams")

        try: os.makedirs(streams_path)
        except: pass
        # Waiting till these directories exist (otherwise potential qstat error)
        while not os.path.isdir(streams_path): pass
        return streams_path

    def _launch_process_group(self, process_commands, streams_path):
        """
        Launches processes defined by process_commands, but only
        executes max_concurrency processes at a time; if a process
        completes and there are still outstanding processes to be
        executed, the next processes are run until max_concurrency is
        reached again.
        """
        processes = {}
        def check_complete_processes(wait=False):
            """
            Returns True if a process completed, False otherwise.
            Optionally allows waiting for better performance (avoids
            sleep-poll cycle if possible).
            """
            result = False
            # list creates copy of keys, as dict is modified in loop
            for proc in list(processes):
                if wait: proc.wait()
                if proc.poll() is not None:
                    # process is done, free up slot
                    self.debug("Process %d exited with code %d."
                               % (processes[proc]['tid'], proc.poll()))
                    processes[proc]['stdout'].close()
                    processes[proc]['stderr'].close()
                    del processes[proc]
                    result = True
            return result

        for cmd, tid in process_commands:
            self.debug("Starting process %d..." % tid)
            job_timestamp = time.strftime('%H%M%S')
            basename = "%s_%s_tid_%d" % (self.batch_name, job_timestamp, tid)
            stdout_handle = open(os.path.join(streams_path, "%s.o.%d"
                                              % (basename, tid)), "wb")
            stderr_handle = open(os.path.join(streams_path, "%s.e.%d"
                                              % (basename, tid)), "wb")
            proc = subprocess.Popen(cmd, stdout=stdout_handle, stderr=stderr_handle)
            processes[proc] = { 'tid' : tid,
                                'stdout' : stdout_handle,
                                'stderr' : stderr_handle }

            if self.max_concurrency:
                # max_concurrency reached, wait until more slots available
                while len(processes) >= self.max_concurrency:
                    if not check_complete_processes(len(processes)==1):
                        time.sleep(0.1)

        # Wait for all processes to complete
        while len(processes) > 0:
            if not check_complete_processes(True):
                time.sleep(0.1)

    def __call__(self):
        """
        Call to start Launcher execution. Typically invoked by
        review_and_launch but may be called directly by the user.
        """
        launchinfo = self._setup_launch()
        streams_path = self._setup_streams_path()
        self.command.finalize(launchinfo)

        self._record_info(launchinfo)

        last_tid = 0
        last_tids = []
        for gid, groupspecs in enumerate(self.args):
            tids = list(range(last_tid, last_tid+len(groupspecs)))
            last_tid += len(groupspecs)
            allcommands = [self.command(
                                self.command._formatter(spec), tid, launchinfo) \
                           for (spec,tid) in zip(groupspecs,tids)]

            self._append_log(list(zip(tids,groupspecs)))

            self.message("Group %d: executing %d processes..." % (gid, len(allcommands)))
            self._launch_process_group(zip(allcommands,tids), streams_path)

            last_tids = tids[:]

            if self.dynamic:
                self.args.update(last_tids, launchinfo)

        self._record_info()
        if self.reduction_fn is not None:
            self.reduction_fn(self._spec_log, self.root_directory)

    def summary(self):
        """
        A succinct summary of the Launcher configuration.  Unlike the
        repr, a summary does not have to be complete but must supply
        key information relevant to the user.
        """
        print("Type: %s" % self.__class__.__name__)
        print("Batch Name: %r" % self.batch_name)
        if self.tag:
            print("Tag: %s" % self.tag)
        print("Root directory: %r" % self.get_root_directory())
        print("Maximum concurrency: %s" % self.max_concurrency)
        if self.description:
            print("Description: %s" % self.description)



class QLauncher(Launcher):
    """
    Launcher that operates with Grid Engine using default arguments
    chosen to be suitable for a typical cluster (tested on
    the Edinburgh University Eddie cluster).

    One of the main features of this class is that it is non-blocking
    - it alway exits shortly after invoking qsub. This means that the
    script is not left running or waiting for long periods of time.

    By convention the standard output and error streams go to the
    corresponding folders in the 'streams' subfolder of the root
    directory - any -o or -e qsub options will be overridden. The job
    name (the -N flag) is specified automatically and any user value
    will be ignored.
    """

    qsub_switches = param.List(default=['-V', '-cwd'], doc = '''
       Specifies the qsub switches (flags without arguments) as a list
       of strings. By default the -V switch is used to exports all
       environment variables in the host environment to the batch job.''')

    qsub_flag_options = param.Dict(default={'-b':'y'}, doc='''
       Specifies qsub flags and their corresponding options as a
       dictionary. Valid values may be strings or lists of string.  If
       a plain Python dictionary is used, the keys arealphanumerically
       sorted, otherwise the dictionary is assumed to be an
       OrderedDict (Python 2.7+ or param.external.OrderedDict) and the
       key ordering will be preserved.

       By default the -b (binary) flag is set to 'y' to allow binaries
       to be directly invoked. Note that the '-' is added to the key
       if missing (to make into a valid flag) so you can specify using
       keywords in the dict constructor: ie. using
       qsub_flag_options=dict(key1=value1, key2=value2, ....)''')

    def __init__(self, batch_name, args, command, **params):
        super(QLauncher, self).__init__(batch_name, args,
                command, **params)

        self._launchinfo = None
        self.last_tids = []
        self._spec_log = []
        self.last_tid = 0
        self.collate_count = 0
        self.spec_iter = iter(self.args)

        self.max_concurrency = None # Inherited


    def _qsub_args(self, override_options, cmd_args, append_options=[]):
        """
        Method to generate Popen style argument list for qsub using
        the qsub_switches and qsub_flag_options parameters. Switches
        are returned first. The qsub_flag_options follow in keys()
        ordered if not a vanilla Python dictionary (ie. a Python 2.7+
        or param.external OrderedDict).  Otherwise the keys are sorted
        alphanumerically. Note that override_options is a list of
        key-value pairs.
        """
        opt_dict = type(self.qsub_flag_options)()
        opt_dict.update(self.qsub_flag_options)
        opt_dict.update(override_options)

        if type(self.qsub_flag_options) == dict:   # Alphanumeric sort if vanilla Python dictionary
            ordered_options = [(k, opt_dict[k]) for k in sorted(opt_dict)]
        else:
            ordered_options =  list(opt_dict.items())

        ordered_options += append_options

        unpacked_groups = [[(k,v) for v in val] if type(val)==list else [(k,val)]
                           for (k,val) in ordered_options]
        unpacked_kvs = [el for group in unpacked_groups for el in group]

        # Adds '-' if missing (eg, keywords in dict constructor) and flattens lists.
        ordered_pairs = [(k,v) if (k[0]=='-') else ('-%s' % (k), v)
                         for (k,v) in unpacked_kvs]
        ordered_options = [[k]+([v] if type(v) == str else list(v)) for (k,v) in ordered_pairs]
        flattened_options = [el for kvs in ordered_options for el in kvs]

        return (['qsub'] + self.qsub_switches
                + flattened_options + [pipes.quote(c) for c in cmd_args])

    def __call__(self):
        """
        Main entry point for the launcher. Collects the static
        information about the launch and sets up the stdout and stderr
        stream output directories. Generates the first call to
        collate_and_launch().
        """
        self._launchinfo = self._setup_launch()
        self.command.finalize(self._launchinfo)
        self.job_timestamp = time.strftime('%H%M%S')

        streams_path = self._setup_streams_path()

        self.qsub_flag_options['-o'] = streams_path
        self.qsub_flag_options['-e'] = streams_path

        self.collate_and_launch()

        self._record_info(self._launchinfo)

    def collate_and_launch(self):
        """
        Method that collates the previous jobs and launches the next
        block of concurrent jobs when using DynamicArgs. This method
        is invoked on initial launch and then subsequently via a
        commandline call (to Python via qsub) to collate the
        previously run jobs and launch the next block of jobs.
        """

        try:   specs = next(self.spec_iter)
        except StopIteration:
            self.qdel_batch()
            if self.reduction_fn is not None:
                self.reduction_fn(self._spec_log, self.root_directory)
            self._record_info()
            return

        tid_specs = [(self.last_tid + i, spec) for (i,spec) in enumerate(specs)]
        self.last_tid += len(specs)
        self._append_log(tid_specs)

        # Updating the argument specifier
        if self.dynamic:
            self.args.update(self.last_tids, self._launchinfo)
        self.last_tids = [tid for (tid,_) in tid_specs]

        output_dir = self.qsub_flag_options['-o']
        error_dir = self.qsub_flag_options['-e']
        self._qsub_block(output_dir, error_dir, tid_specs)

        # Pickle launcher before exit if necessary.
        if self.dynamic or (self.reduction_fn is not None):
            pickle_path = os.path.join(self.root_directory, 'qlauncher.pkl')
            pickle.dump(self, open(pickle_path,'wb'))

    def _qsub_collate_and_launch(self, output_dir, error_dir, job_names):
        """
        The method that actually runs qsub to invoke the python
        process with the necessary commands to trigger the next
        collation step and next block of jobs.
        """

        job_name = "%s_%s_collate_%d" % (self.batch_name,
                                         self.job_timestamp,
                                         self.collate_count)

        overrides = [("-e",error_dir), ('-N',job_name), ("-o",output_dir),
                     ('-hold_jid',','.join(job_names))]

        resume_cmds =["import os, pickle, lancet",
                      ("pickle_path = os.path.join(%r, 'qlauncher.pkl')"
                       % self.root_directory),
                      "launcher = pickle.load(open(pickle_path,'rb'))",
                      "launcher.collate_and_launch()"]

        cmd_args = [self.command.executable,
                    '-c', ';'.join(resume_cmds)]
        popen_args = self._qsub_args(overrides, cmd_args)

        p = subprocess.Popen(popen_args, stdout=subprocess.PIPE)
        (stdout, stderr) = p.communicate()

        self.debug(stdout)
        if p.poll() != 0:
            raise EnvironmentError("qsub command exit with code: %d" % p.poll())

        self.collate_count += 1
        self.message("Invoked qsub for next batch.")
        return job_name

    def _qsub_block(self, output_dir, error_dir, tid_specs):
        """
        This method handles static argument specifiers and cases where
        the dynamic specifiers cannot be queued before the arguments
        are known.
        """
        processes = []
        job_names = []

        for (tid, spec) in tid_specs:
            job_name = "%s_%s_tid_%d" % (self.batch_name, self.job_timestamp, tid)
            job_names.append(job_name)
            cmd_args = self.command(
                    self.command._formatter(spec),
                    tid, self._launchinfo)

            popen_args = self._qsub_args([("-e",error_dir), ('-N',job_name), ("-o",output_dir)],
                                        cmd_args)
            p = subprocess.Popen(popen_args, stdout=subprocess.PIPE)
            (stdout, stderr) = p.communicate()

            self.debug(stdout)
            if p.poll() != 0:
                raise EnvironmentError("qsub command exit with code: %d" % p.poll())

            processes.append(p)

        self.message("Invoked qsub for %d commands" % len(processes))
        if (self.reduction_fn is not None) or self.dynamic:
            self._qsub_collate_and_launch(output_dir, error_dir, job_names)


    def qdel_batch(self):
        """
        Runs qdel command to remove all remaining queued jobs using
        the <batch_name>* pattern . Necessary when StopIteration is
        raised with scheduled jobs left on the queue.
        Returns exit-code of qdel.
        """
        p = subprocess.Popen(['qdel', '%s_%s*' % (self.batch_name,
                                                  self.job_timestamp)],
                             stdout=subprocess.PIPE)
        (stdout, stderr) = p.communicate()
        return p.poll()


class ScriptLauncher(Launcher):
    """
    Script-based launcher. Calls a script with a path to a JSON file containing
    process group job options. This easily supports more environment-specific
    job-submission schemes without having to create a new Launcher every time.
    """
    script_path = param.String(default=os.path.join(os.getcwd(), 'launch_process_group.py'), doc='''
        Path to script which is called for every group, with JSON file,
        batch_name, number of commands for this group and max_concurrency as
        arguments.''')

    json_name = param.String(default='processes_%s.json', doc='''
        Name of the JSON file output per process group.''')

    def __init__(self, batch_name, args, command, **params):
        super(ScriptLauncher, self).__init__(batch_name, args,
                command, **params)

    def _launch_process_group(self, process_commands, streams_path):
        """
        Aggregates all process_commands and the designated output files into a
        list, and outputs it as JSON, after which the wrapper script is called.
        """
        processes = []
        for cmd, tid in process_commands:
            job_timestamp = time.strftime('%H%M%S')
            basename = "%s_%s_tid_%d" % (self.batch_name, job_timestamp, tid)
            stdout_path = os.path.join(streams_path, "%s.o.%d" % (basename, tid))
            stderr_path = os.path.join(streams_path, "%s.e.%d" % (basename, tid))
            process = { 'tid' : tid,
                        'cmd' : cmd,
                        'stdout' : stdout_path,
                        'stderr' : stderr_path }
            processes.append(process)

        # To make the JSON filename unique per group, we use the last tid in
        # this group.
        json_path = os.path.join(self.root_directory, self.json_name % (tid))
        with open(json_path, 'w') as json_file:
            json.dump(processes, json_file, sort_keys=True, indent=4)

        p = subprocess.Popen([self.script_path, json_path, self.batch_name,
                              str(len(processes)), str(self.max_concurrency)])
        if p.wait() != 0:
            raise EnvironmentError("Script command exit with code: %d" % p.poll())

#===============#
# Launch Helper #
#===============#

class review_and_launch(core.PrettyPrinted, param.Parameterized):
    """
    A helper decorator that always checks for consistency and can
    prompt the user for a full review of the launch configuration.
    """

    output_directory = param.String(default='.', doc='''
         The output directory - the directory that will contain all
         the root directories for the individual launches.''')

    review = param.Boolean(default=True, doc='''
         Whether or not to perform a detailed review of the launch.''')

    launch_args = param.ClassSelector(default=None, allow_None=True,
        class_=core.Args, doc= '''An optional argument specifier to
        parameterise lancet, allowing multi-launch scripts.  For
        instance, this may be useful for collecting statistics over
        runs that are not deterministic or are affected by a random
        input seed.''')

    launch_fn = param.Callable(doc='''The function that is to be decorated.''')

    def __init__(self, **params):
        self._pprint_args = ([],[],None,{})
        super(review_and_launch, self).__init__(**params)
        self.pprint_args(['output_directory', 'launch_fn'],
                         ['review', 'launch_args'])

    def cross_check_launchers(self, launchers):
        """
        Performs consistency checks across all the launchers.
        """
        if len(launchers) == 0: raise Exception('Empty launcher list')
        timestamps = [launcher.timestamp for launcher in launchers]

        if not all(timestamps[0] == tstamp for tstamp in timestamps):
            raise Exception("Launcher timestamps not all equal. "
                            "Consider setting timestamp explicitly.")

        root_directories = []
        for launcher in launchers:
            command = launcher.command
            args = launcher.args
            command.verify(args)
            root_directory = launcher.get_root_directory()
            if os.path.isdir(root_directory):
                raise Exception("Root directory already exists: %r" % root_directory)
            if root_directory in root_directories:
                raise Exception("Each launcher requires a unique root directory")
            root_directories.append(root_directory)

    def __call__(self, fn=None):

        # On first call, simply wrap the provided launch function.
        if fn is not None:
            self.launch_fn = fn
            return self

        # Calling the wrapped function with appropriate arguments as
        # supplied by launch_args.
        kwargs = self.launch_args.specs if self.launch_args else [{}]
        launchers = [self.launch_fn(**kwargs[0])]
        if self.launch_args is not None:
            launchers += [self.launch_fn(**kws) for kws in kwargs[1:]]

        current_timestamp = tuple(time.localtime())

        # Across all the launchers...
        for launcher in launchers:
            # Ensure a shared timestamp throughout
            launcher.timestamp = current_timestamp
            # Set the output directory appropriately
            if self.output_directory is not None:
                launcher.output_directory = self.output_directory

        # Cross check the launchers
        self.cross_check_launchers(launchers)

        if not self.review:
            return self._launch_all(launchers)
        else:
            return self._review_all(launchers)

    def _launch_all(self, launchers):
        """
        Launches all available launchers.
        """
        for launcher in launchers:
            print("== Launching  %s ==" % launcher.batch_name)
            launcher()
        return True

    def _review_all(self, launchers):
        """
        Runs the review process for all the launchers.
        """
        # Run review of launch args if necessary
        if self.launch_args is not None:
            proceed = self.review_args(self.launch_args,
                                       show_repr=True,
                                       heading='Meta Arguments')
            if not proceed: return False

        reviewers = [self.review_args,
                     self.review_command,
                     self.review_launcher]

        for (count, launcher) in enumerate(launchers):

            # Run reviews for all launchers if desired...
            if not all(reviewer(launcher) for reviewer in reviewers):
                print("\n == Aborting launch ==")
                return False
            # But allow the user to skip these extra reviews
            if len(launchers)!= 1 and count < len(launchers)-1:
                skip_remaining = self.input_options(['Y', 'n','quit'],
                                 '\nSkip remaining reviews?', default='y')

                if skip_remaining == 'y':          break
                elif skip_remaining == 'quit':     return False

        if self.input_options(['y','N'], 'Execute?', default='n') != 'y':
            return False
        else:
            return self._launch_all(launchers)

    def review_launcher(self, launcher):
        launcher_name = launcher.__class__.__name__
        print('%s\n' % self.summary_heading(launcher_name))
        launcher.summary()
        print('')
        if self.input_options(['Y','n'],
                              '\nShow complete launch repr?', default='y') == 'y':
            print("\n%s\n" % launcher)
        return True

    def review_args(self, obj, show_repr=False, heading='Arguments'):
        """
        Reviews the given argument specification. Can review the
        meta-arguments (launch_args) or the arguments themselves.
        """
        args = obj.args if isinstance(obj, Launcher) else obj
        print('\n%s\n' % self.summary_heading(heading))
        args.summary()
        if show_repr: print("\n%s\n" % args)
        response = self.input_options(['y', 'N','quit'],
                '\nShow available argument specifier entries?', default='n')
        if response == 'quit': return False
        if response == 'y':  args.show()
        print('')
        return True

    def review_command(self, launcher):

        command = launcher.command
        template_name = command.__class__.__name__
        print('%s\n' % self.summary_heading(template_name))
        command.summary()
        print("\n")
        response = self.input_options(['y', 'N','quit','save'],
                                      '\nShow available command entries?',
                                      default='n')

        args = launcher.args
        if response == 'quit': return False
        elif response == 'y':
            command.show(args)
        elif response == 'save':
            fname = input('Filename: ').replace(' ','_')
            with open(os.path.abspath(fname),'w') as f:
                command.show(args, file_handle=f)
        print('')
        return True

    def summary_heading(self, text, car='=', carvert='|'):
        text = text + " Summary"
        length=len(text)+4
        return '%s\n%s %s %s\n%s' % (car*length, carvert, text,
                                     carvert, car*length)

    def input_options(self, options, prompt='Select option', default=None):
        """
        Helper to prompt the user for input on the commandline.
        """
        check_options = [x.lower() for x in options]
        while True:
            response = input('%s [%s]: ' % (prompt, ', '.join(options))).lower()
            if response in check_options: return response.strip()
            elif response == '' and default is not None:
                return default.lower().strip()
