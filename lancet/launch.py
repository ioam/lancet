#
# Lancet launchers and launch helpers
#

import os, sys, time, pipes, subprocess, types
import fnmatch
import json, pickle
import logging

import param

import lancet.core as core
from lancet.dynamic import DynamicArgs

#===================#
# Commands Template #
#===================#

class CommandTemplate(param.Parameterized):
    """
    A command template is a way of converting the dictionaries
    returned by argument specifiers into a particular command. When
    called with an argument specifier, a command template returns a
    list of strings corresponding to a subprocess Popen argument list.

    __call__(self, spec, tid=None, info={}):

    All CommandTemplates must be callable. The tid argument is the
    task id and info is a dictionary of run-time information supplied
    by the launcher. See the _setup_launch method of Launcher to see
    details about the launch information supplied.
    """

    executable = param.String(default='python', constant=True, doc='''
        The executable that is to be run by this
        CommandTemplate. Unless the executable is a standard command
        expected on the system path, this should be an absolute
        path. By default this invokes python or the python environment
        used to invoke the CommandTemplate (eg. topographica).''')

    do_format = param.Boolean(default=True, doc= '''
        Set to True to receive input arguments as formatted strings,
        False for the raw unformatted objects.''')

    def __init__(self, executable=None, **kwargs):
        if executable is None:
            executable = sys.executable
        super(CommandTemplate,self).__init__(executable=executable, **kwargs)

    def __call__(self, spec, tid=None, info={}):
        """
        Formats a single argument specification supplied as a
        dictionary of argument name/value pairs. The info dictionary
        contains launch information as defined in the _setup_launch
        method of Launcher.
        """
        raise NotImplementedError

    def _formatter(self, spec):
        if self.do_format: return core.BaseArgs.spec_formatter(spec)
        else             : return spec

    def validate_arguments(self, args):
        """
        Allows a final check that ensures valid arguments have been
        passed before launch. Allows the constant and varying_keys to
        be be checked and can inspect the specs attribute if an
        instance of Args. If invalid, raise an Exception with the
        appropriate error message.
        """
        return

    def show(self, arg_specifier, file_handle=sys.stdout, **kwargs):
        full_string = ''
        info = {'root_directory':     '<root_directory>',
                'batch_name':         '<batch_name>',
                'batch_tag':          '<batch_tag>',
                'batch_description':  '<batch_description>',
                'timestamp':          tuple(time.localtime()),
                'timestamp_format':   '<timestamp_format>',
                'varying_keys':       arg_specifier.varying_keys,
                'constant_keys':      arg_specifier.constant_keys,
                'constant_items':     arg_specifier.constant_items}

        quoted_cmds = [ subprocess.list2cmdline(
                [el for el in self(self._formatter(s),'<tid>',info)])
                        for s in arg_specifier.specs]

        cmd_lines = ['%d: %s\n' % (i, qcmds) for (i,qcmds)
                     in enumerate(quoted_cmds)]
        full_string += ''.join(cmd_lines)

        file_handle.write(full_string)
        file_handle.flush()


class UnixCommand(CommandTemplate):
    """
    A generic CommandTemplate useable with most Unix commands. By
    default, follows the GNU coding convention for commandline
    arguments.
    """

    expansions = param.Dict(default={}, doc="""
        Allows extension of the specification that supports functions
        that expand to valid argument values.  If a function is used,
        it must have the signature (spec, info, tid). A typical usage
        for a function value is to build a valid output filename given
        the contrext.

        Three such function are provided as classmethods:
        'root_directory', 'long_filename' and 'expand'.""")

    posargs = param.List(default=[], doc="""
       The list of positional argument keys. Positional arguments are
       always supplied at the end of a command in the order given.""")

    long_prefix = param.String(default='--',  doc="""
       Although the double dash is a GNU coding convention, some
       applications use single dashes for long options.""")

    def __init__(self, executable, **kwargs):
        super(UnixCommand,self).__init__(executable = executable,
                                         do_format=False,
                                         **kwargs)

    def __call__(self, spec, tid=None, info={}):
        # Function expansions are called here.
        expanded = {}
        for (k,v) in self.expansions.items():
            if isinstance(v, types.FunctionType):
                expanded[k] = v(spec, info, tid)
            else:
                expanded[k] = v

        expanded.update(spec)
        expanded = core.BaseArgs.spec_formatter(expanded)

        options = []
        for (k, v) in expanded.items():
            if k in self.posargs or spec[k] == False:
                continue
            options.append('%s%s' % (self.long_prefix if len(k) > 1 else '-', k))
            if spec[k] != True:
                options.append(v)

        posargs = [expanded[parg] if (parg in expanded) else parg(spec, info, tid)
                   for parg in self.posargs]
        return [self.executable] + options + posargs

    @classmethod
    def root_directory(cls):
        """
        Supplies the root_directory to a command.
        """
        def _expander(spec, info, tid):
            return  info['root_directory']
        return _expander

    @classmethod
    def long_filename(cls, extension, excluding=[]):
        """
        Generates a long filename based on the input arguments in the
        root directory with the given extension. Ignores constant
        items.
        """
        def _expander(spec, info, tid):
            root_dir = info['root_directory']
            params = [('tid' , tid)] + [(k,v) for  (k,v) in spec.items()
                                        if k in info['varying_keys'] and k not in excluding]
            basename = '_'.join('%s=%s' % (k,v) for (k,v) in sorted(params))
            return os.path.join(root_dir, '%s_%s%s' % (info['batch_name'], basename, extension))
        return _expander

    @classmethod
    def expand(cls, template):
        """
        Takes a new-style format string template and expands it out
        using the keys in the spec, info and tid.
        """
        def _expander(spec, info, tid):
            all_params = {'tid' : tid}
            all_params.update(spec)
            all_params.update(info)
            return template.format(**all_params)
        return _expander


#===========#
# Launchers #
#===========#

class Launcher(param.Parameterized):
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

    arg_specifier = param.ClassSelector(core.BaseArgs, constant=True, doc='''
       The specifier used to generate the varying parameters for the tasks.''')

    command_template = param.ClassSelector(CommandTemplate, constant=True, doc='''
       The command template used to generate the commands for the current tasks.''')

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

    output_directory = param.String(default='.', doc="""
         The output directory - the directory that will contain all
         the root directories for the individual launches.""")

    subdir = param.List(default=[], doc="""
      A list of subdirectory names that allows custom organization
      within the output directory before the root directory.""")


    def __init__(self, batch_name, arg_specifier, command_template, **kwargs):
        super(Launcher,self).__init__(arg_specifier=arg_specifier,
                                      command_template = command_template,
                                      **kwargs)
        self.batch_name = batch_name
        self._spec_log = []
        if self.timestamp == (0,)*9:
            self.timestamp = tuple(time.localtime())

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

    def append_log(self, specs):
        """
        The log contains the tids and corresponding specifications
        used during launch with the specifications in JSON format.
        """
        self._spec_log += specs # This should be removed
        log_path = os.path.join(self.root_directory, ("%s.log" % self.batch_name))
        core.Log.write_log(log_path, [spec for (_, spec) in specs], allow_append=True)

    def record_info(self, setup_info=None):
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

        return {'root_directory':    self.root_directory,
                'timestamp':         self.timestamp,
                'timestamp_format':  self.timestamp_format,
                'varying_keys':      self.arg_specifier.varying_keys,
                'constant_keys':     self.arg_specifier.constant_keys,
                'constant_items':     self.arg_specifier.constant_items,
                'batch_name':        self.batch_name,
                'batch_tag':         self.tag,
                'batch_description': self.description }

    def _setup_streams_path(self):
        streams_path = os.path.join(self.root_directory, "streams")

        try: os.makedirs(streams_path)
        except: pass
        # Waiting till these directories exist (otherwise potential qstat error)
        while not os.path.isdir(streams_path): pass
        return streams_path

    def launch_process_group(self, process_commands, streams_path):
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
            for proc in list(processes): # make list (copy) of keys, as dict is modified during iteration
                if wait: proc.wait()
                if proc.poll() is not None:
                    # process is done, free up slot
                    logging.debug("Process %d exited with code %d." % (processes[proc]['tid'], proc.poll()))
                    processes[proc]['stdout'].close()
                    processes[proc]['stderr'].close()
                    del processes[proc]
                    result = True
            return result

        for cmd, tid in process_commands:
            logging.debug("Starting process %d..." % tid)
            stdout_handle = open(os.path.join(streams_path, "%s.o.%d" % (self.batch_name, tid)), "wb")
            stderr_handle = open(os.path.join(streams_path, "%s.e.%d" % (self.batch_name, tid)), "wb")
            proc = subprocess.Popen(cmd, stdout=stdout_handle, stderr=stderr_handle)
            processes[proc] = { 'tid' : tid, 'stdout' : stdout_handle, 'stderr' : stderr_handle }

            if self.max_concurrency:
                # max_concurrency reached, wait until more slots available
                while len(processes) >= self.max_concurrency:
                    if not check_complete_processes(len(processes)==1):
                        time.sleep(0.1)

        # Wait for all processes to complete
        while len(processes) > 0:
            if not check_complete_processes(True):
                time.sleep(0.1)

    def launch(self):
        """
        The method that starts Launcher execution. Typically called by
        a launch helper.  This could be called directly by the user.
        """
        launchinfo = self._setup_launch()
        streams_path = self._setup_streams_path()

        self.record_info(launchinfo)

        last_tid = 0
        last_tids = []
        for gid, groupspecs in enumerate(self.arg_specifier):
            tids = list(range(last_tid, last_tid+len(groupspecs)))
            last_tid += len(groupspecs)
            allcommands = [self.command_template(
                                self.command_template._formatter(spec), tid, launchinfo) \
                           for (spec,tid) in zip(groupspecs,tids)]

            self.append_log(list(zip(tids,groupspecs)))

            logging.info("Group %d: executing %d processes..." % (gid, len(allcommands)))
            self.launch_process_group(zip(allcommands,tids), streams_path)

            last_tids = tids[:]

            if isinstance(self.arg_specifier, DynamicArgs):
                self.arg_specifier.update(last_tids, launchinfo)

        self.record_info()
        if self.reduction_fn is not None:
            self.reduction_fn(self._spec_log, self.root_directory)

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

    qsub_switches = param.List(default=['-V', '-cwd'], doc = """
       Specifies the qsub switches (flags without arguments) as a list
       of strings. By default the -V switch is used to exports all
       environment variables in the host environment to the batch job.""")

    qsub_flag_options = param.Dict(default={'-b':'y'}, doc="""
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
       qsub_flag_options=dict(key1=value1, key2=value2, ....)""")

    def __init__(self, batch_name, arg_specifier, command_template, **kwargs):
        super(QLauncher, self).__init__(batch_name, arg_specifier,
                command_template, **kwargs)

        self._launchinfo = None
        self.schedule = None
        self.last_tids = []
        self._spec_log = []
        self.last_tid = 0
        self.last_scheduled_tid = 0
        self.collate_count = 0
        self.spec_iter = iter(self.arg_specifier)

        self.max_concurrency = None # Inherited

        # The necessary conditions for reserving jobs before specification known.
        self.is_dynamic_qsub = all([isinstance(self.arg_specifier, DynamicArgs),
                                    hasattr(self.arg_specifier, 'schedule'),
                                    hasattr(self.command_template,   'queue'),
                                    hasattr(self.command_template,   'specify')])

    def qsub_args(self, override_options, cmd_args, append_options=[]):
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

    def launch(self):
        """
        Main entry point for the launcher. Collects the static
        information about the launch and sets up the stdout and stderr
        stream output directories. Generates the first call to
        collate_and_launch().
        """
        self._launchinfo = self._setup_launch()
        self.job_timestamp = time.strftime('%H%M%S')

        streams_path = self._setup_streams_path()

        self.qsub_flag_options['-o'] = streams_path
        self.qsub_flag_options['-e'] = streams_path

        self.collate_and_launch()

        self.record_info(self._launchinfo)

    def collate_and_launch(self):
        """
        Method that collates the previous jobs and launches the next
        block of concurrent jobs. The launch type can be either static
        or dynamic (using schedule, queue and specify for dynamic
        argument specifiers).  This method is invoked on initial
        launch and then subsequently via the commandline to collate
        the previously run jobs and launching the next block of jobs.
        """

        try:   specs = next(self.spec_iter)
        except StopIteration:
            self.qdel_batch()
            if self.reduction_fn is not None:
                self.reduction_fn(self._spec_log, self.root_directory)
            self.record_info()
            return

        tid_specs = [(self.last_tid + i, spec) for (i,spec) in enumerate(specs)]
        self.last_tid += len(specs)
        self.append_log(tid_specs)

        # Updating the argument specifier
        if isinstance(self.arg_specifier,DynamicArgs):
            self.arg_specifier.update(self.last_tids, self._launchinfo)
        self.last_tids = [tid for (tid,_) in tid_specs]

        output_dir = self.qsub_flag_options['-o']
        error_dir = self.qsub_flag_options['-e']
        if self.is_dynamic_qsub: self.dynamic_qsub(output_dir, error_dir, tid_specs)
        else:            self.static_qsub(output_dir, error_dir, tid_specs)

        # Pickle launcher before exit if necessary.
        if isinstance(self.arg_specifier,DynamicArgs) or (self.reduction_fn is not None):
            pickle_path = os.path.join(self.root_directory, 'qlauncher.pkl')
            pickle.dump(self, open(pickle_path,'wb'))

    def qsub_collate_and_launch(self, output_dir, error_dir, job_names):
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

        cmd_args = [self.command_template.executable,
                    '-c', ';'.join(resume_cmds)]
        popen_args = self.qsub_args(overrides, cmd_args)

        p = subprocess.Popen(popen_args, stdout=subprocess.PIPE)
        (stdout, stderr) = p.communicate()

        self.collate_count += 1
        logging.debug(stdout)
        logging.info("Invoked qsub for next batch.")
        return job_name

    def static_qsub(self, output_dir, error_dir, tid_specs):
        """
        This method handles static argument specifiers and cases where
        the dynamic specifiers cannot be queued before the arguments
        are known.
        """
        processes = []
        job_names = []

        for (tid, spec) in tid_specs:
            job_name = "%s_%s_job_%d" % (self.batch_name, self.job_timestamp, tid)
            job_names.append(job_name)
            cmd_args = self.command_template(
                    self.command_template._formatter(spec),
                    tid, self._launchinfo)

            popen_args = self.qsub_args([("-e",error_dir), ('-N',job_name), ("-o",output_dir)],
                                        cmd_args)
            p = subprocess.Popen(popen_args, stdout=subprocess.PIPE)
            (stdout, stderr) = p.communicate()
            logging.debug(stdout)
            processes.append(p)

        logging.info("Invoked qsub for %d commands" % len(processes))
        if self.reduction_fn is not None:
            self.qsub_collate_and_launch(output_dir, error_dir, job_names)

    def dynamic_qsub(self, output_dir, error_dir, tid_specs):
        """
        This method handles dynamic argument specifiers where the
        dynamic argument specifier can be queued before the arguments
        are computed.
        """

        # Write out the specification files in anticipation of execution
        for (tid, spec) in tid_specs:
            self.command_template.specify(
                    self.command_template._formatter(spec),
                    tid, self._launchinfo)

        # If schedule is empty (or on first initialization)...
        if (self.schedule == []) or (self.schedule is None):
            self.schedule = self.arg_specifier.schedule()
            assert len(tid_specs)== self.schedule[0], "Number of specs don't match schedule!"

            # Generating the scheduled tasks (ie the queue commands)
            collate_name = None
            for batch_size in self.schedule:
                schedule_tids = [tid + self.last_scheduled_tid for tid in range(batch_size) ]
                schedule_tasks = [(tid, self.command_template.queue(tid, self._launchinfo)) for
                                      tid in schedule_tids]

                # Queueing with the scheduled tasks with appropriate job id dependencies
                hold_jid_cmd = []
                group_names = []

                for (tid, schedule_task) in schedule_tasks:
                    job_name = "%s_%s_job_%d" % (self.batch_name, self.job_timestamp, tid)
                    overrides = [("-e",error_dir), ('-N',job_name), ("-o",output_dir)]
                    if collate_name is not None: overrides += [('-hold_jid', collate_name)]
                    popen_args = self.qsub_args(overrides, schedule_task)
                    p = subprocess.Popen(popen_args, stdout=subprocess.PIPE)
                    (stdout, stderr) = p.communicate()
                    group_names.append(job_name)

                collate_name = self.qsub_collate_and_launch(output_dir, error_dir, group_names)
                self.last_scheduled_tid += batch_size

            # Popping the currently specified tasks off the schedule
            self.schedule = self.schedule[1:]

    def qdel_batch(self):
        """
        Runs qdel command to remove all remaining queued jobs using
        the <batch_name>* pattern . Necessary when StopIteration is
        raised with scheduled jobs left on the queue.
        """
        p = subprocess.Popen(['qdel', '%s_%s*' % (self.batch_name, self.job_timestamp)],
                             stdout=subprocess.PIPE)
        (stdout, stderr) = p.communicate()

#===============#
# Launch Helper #
#===============#

class review_and_launch(param.Parameterized):
    """
    A helper decorator that always checks for consistency and can
    prompt the user for a full review of the launch configuration.
    """

    output_directory = param.String(default='.', doc="""
         The output directory - the directory that will contain all
         the root directories for the individual launches.""")

    review = param.Boolean(default=True, doc="""
         Whether or not to perform a detailed review of the launch.""")

    launch_args = param.ClassSelector(default=None, allow_None=True, class_=core.Args,
         doc= """An optional argument specifier to parameterise
                 lancet, allowing multi-launch scripts.  For instance,
                 this may be useful for collecting statistics over
                 runs that are not deterministic or are affected by a
                 random input seed.""")

    launch_fn = param.Callable(doc="""The function that is to be applied.""")

    def __init__(self, output_directory='.', **kwargs):

        super(review_and_launch, self).__init__( output_directory=output_directory,
                                                 **kwargs)

    def section(self, text, car='=', carvert='|'):
        length=len(text)+4
        return '%s\n%s %s %s\n%s' % (car*length, carvert, text,
                                     carvert, car*length)

    def input_options(self, options, prompt='Select option', default=None):
        """
        Helper to prompt the user for input on the commandline.
        """
        check_options = [x.lower() for x in options]
        while True:
            response = raw_input('%s [%s]: ' % (prompt, ', '.join(options))).lower()
            if response in check_options: return response.strip()
            elif response == '' and default is not None:
                return default.lower().strip()

    def cross_check_launchers(self, launchers):
        """
        Performs consistency checks across all the launchers.
        """
        if len(launchers) == 0: raise Exception('Empty launcher list')
        timestamps = [launcher.timestamp for launcher in launchers]

        if not all(timestamps[0] == tstamp for tstamp in timestamps):
            raise Exception("Launcher timestamps not all equal. "
                            "Consider setting timestamp explicitly.")

        # Needs to be made compatible with the subdir option of Launcher
        # if len(set(batch_names)) != len(launchers):
        #     raise Exception('Each launcher requires a unique batch name.')

        for launcher in launchers:
            command_template = launcher.command_template
            arg_specifier = launcher.arg_specifier
            command_template.validate_arguments(arg_specifier)

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
            launcher.__class__.timestamp = current_timestamp
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
            launcher.launch()
        return True

    def _review_all(self, launchers):
        """
        Runs the review process for all the launchers.
        """
        # Run review of launch args if necessary
        if self.launch_args is not None:
            proceed = self.review_args(self.launch_args,
                                       heading='Lancet Meta Parameters')
            if not proceed: return False

        reviewers = [self.review_launcher,
                     self.review_args,
                     self.review_command_template]

        for (count, launcher) in enumerate(launchers):

            if not all(reviewer(launcher) for reviewer in reviewers):
                print("\n == Aborting launch ==")
                return False

            if len(launchers)!= 1 and count < len(launchers)-1:
                skip_remaining = self.input_options(['Y', 'n','quit'],
                                 '\nSkip remaining reviews?', default='y')
                if skip_remaining == 'quit':
                    return False
                if skip_remaining == 'y':
                    break

        if self.input_options(['y','N'], 'Execute?', default='n') != 'y':
            return False
        else:
            return self._launch_all(launchers)

    def review_launcher(self, launcher):
        command_template = launcher.command_template
        print(self.section('Launcher'))
        print("Type: %s" % launcher.__class__.__name__)
        print("Batch Name: %s" % launcher.batch_name)
        print("Command executable: %s" % command_template.executable)
        root_directory = launcher.get_root_directory()
        print("Root directory: %s" % root_directory)
        print("Maximum concurrency: %s" % launcher.max_concurrency)

        description = '<No description>' if launcher.description == '' else launcher.description
        tag = '<No tag>' if launcher.tag == '' else launcher.tag
        print("Description: %s" % description)
        print("Tag: %s" % tag)
        print
        return True

    def review_args(self, obj, heading='Argument Specification'):
        """
        Input argument obj can be a Launcher or an ArgSpec object.
        """
        arg_specifier = obj.arg_specifier if isinstance(obj, Launcher) else obj
        print(self.section(heading))
        print("Type: %s (dynamic=%s)" %
              (arg_specifier.__class__.__name__, isinstance(arg_specifier,DynamicArgs)))
        print("Varying Keys: %s" % arg_specifier.varying_keys)
        items = '\n'.join(['%s = %r' % (k,v) for (k,v) in arg_specifier.constant_items])
        print("Constant Items:\n\n%s\n" % items)
        print("Definition:\n%s" % arg_specifier)

        response = self.input_options(['y', 'N','quit'],
                '\nShow available argument specifier entries?', default='n')
        if response == 'quit': return False
        if response == 'y':  arg_specifier.show()
        print
        return True

    def review_command_template(self, launcher):
        command_template = launcher.command_template
        arg_specifier = launcher.arg_specifier
        print(self.section(command_template.__class__.__name__))
        if isinstance(launcher, QLauncher) and launcher.is_dynamic_qsub:
            command_template.show(arg_specifier, queue_cmd_only=True)

        response = self.input_options(['y', 'N','quit','save'],
                                      '\nShow available command entries?', default='n')
        if response == 'quit': return False
        if response == 'y':
            command_template.show(arg_specifier)
        if response == 'save':
            fname = raw_input('Filename: ').replace(' ','_')
            with open(os.path.abspath(fname),'w') as f:
                command_template.show(arg_specifier, file_handle=f)
        print
        return True

    def __repr__(self):
        arg_list = ['%r' % self.output_directory,
                    'launch_args=%r' % self.launch_args if self.launch_args else None,
                    'review=False' if not self.review else None]
        arg_str = ','.join(el for el in arg_list if el is not None)
        return 'review_and_launch(%s)' % arg_str

    def __str__(self):
        arg_list = ['output_directory=%r' % self.output_directory,
                    'launch_args=%s' % self.launch_args._pprint(level=2) if self.launch_args else None,
                    'review=False' if not self.review else None]
        arg_str = ',\n   '.join(el for el in arg_list if el is not None)
        return 'review_and_launch(\n   %s\n)' % arg_str
