"""
Lancet consists of three fundamental class types: Argument Specifiers, Command
Templates and Launchers. Launch Helpers are useful helper functions for using
Lancet.

Argument Specifiers
-------------------

Argument specifiers are intended to be clear, succinct and composable way of
specifying large parameter sets. High-dimensional parameter sets are typical of
large-scale scientific models and can make managing such models and simulations
difficult. Using argument specifiers, you can document how the parameters of
your model vary across runs without long, flat lists of arguments or requiring
deeply nested loops.

Argument specifiers can be freely intermixed with Python code, simplifying the
use of scientific software with a Python interface. They also invoke commandline
programs using command template objects to build commands and launchers to
execute them. In this way they can help simplify the management of simulation,
analysis and visualisation tools with Python.

Typical usage includes specification of constant arguments, description of
parameter ranges or even search procedures (such as hillclimbing, bisection
optimisation etc). Argument specifiers can compose together using Cartesian
Products or Concatenation to express huge numbers of argument sets concisely.

Command Templates
------------------

When working with external tools, a command template is needed to turn an
argument specifier into an executable command. These objects are designed to be
customisable with sensible defaults to handle as much boilerplate as
possible. They may require argument specifiers with a certain structure (certain
argument names and value types) to operate correctly.

Launchers
---------

A launcher is designed to execute commands on a given computational platform,
making use of as much concurrency where possible. The default Launcher class
executes commands locally while the QLauncher class is designed to launch
commands on Sun Grid Engine clusters.

Launch Helpers
--------------

These helper functions help verify correct usage of lancet for specific
use-cases. They check for consistent useage automatically and offer an in-depth
review of all settings specified before launch. The goal is to help users
identify mistakes early before consuming computational time and resources.
"""

import os, sys, copy, re, glob, json, pickle, fnmatch
import time, pipes, subprocess
import logging

import numpy as np
import itertools
from collections import defaultdict

import param

float_types = [float] + np.sctypes['float']
def identityfn(x): return x
def fp_repr(x):    return str(x) if (type(x) in float_types) else repr(x)

def set_fp_precision(value):
    """
    Function to set the floating precision across lancet.
    """
    BaseArgs.set_default('fp_precision', value)

#=====================#
# Argument Specifiers #
#=====================#

class BaseArgs(param.Parameterized):
    """
    The base class for all argument specifiers. Argument specifiers implement
    Python's iterator protocol, returning arguments on each call to next(). Note
    that these objects should not be written as generators as they often need to
    be pickled.

    On each iteration, an argument specifier returns a list of
    dictionaries. Each key-value pair assigns a value to an argument and each
    dictionary defines a set of arguments.  The dictionaries in the list
    correspond to the currently available argument sets. Subsequent iterations
    may return further arguments for specifiers that were waiting for runtime
    feedback to continue (eg. hillclimbing).

    Such argument specifiers that require feedback are dynamic: in these cases
    dynamic=True and the state of the specifier must be updated between
    iterations via update(). Whenever possible, the schedule() method specifies
    the expected number of argument dictionaries on future iterations. Such
    expected counts may be incorrect - for example, an initial hillclimbing seed
    may already happen to be very close to a local optimum. Note that the
    update() and schedule() methods do not need to be implemented if
    dynamic=False.

    To help enforce a declarative style, public parameters are declared constant
    and cannot be mutated after initialisation. Argument specifiers need to be
    declared correctly where they are used.

    The varying_keys and constant_keys methods makes the distinction between
    arguments that are manipulated between runs and the ones that stay constant.
    The values of the arguments that vary is often of more interest than the
    values of the arguments held constant and varying arguments are sorted by
    how fast they vary.
    """

    dynamic = param.Boolean(default=False, constant=True, doc='''
            Flag to indicate whether the argument specifier needs to have its
            state updated via the update() method between iterations. In other
            words, if the arguments cannot be known ahead of time (eg. parameter
            search), dynamic must be set to True.''')

    fp_precision = param.Integer(default=4,  constant=True, doc='''
         The floating point precision to use for floating point values.  Unlike
         other basic Python types, floats need care with their representation as
         you only want to display up to the precision actually specified.''')

    def __init__(self,   **params):
        super(BaseArgs,self).__init__(**params)

    def __iter__(self): return self

    def spec_formatter(self, spec):
        " Formats the elements of an argument set appropriately"
        return dict((k, str(v)) for (k,v) in spec.items())

    def constant_keys(self):
        """
        Returns the list of parameter names whose values are constant as the
        argument specifier is iterated.  Note that constant_keys() +
        varying_keys() should span the entire set of keys.
        """
        raise NotImplementedError

    def varying_keys(self):
        """
        Returns the list of parameters whose values vary as the argument
        specifier is iterated.  Whenever it is possible, keys should be sorted
        from those slowest to faster varying and sorted alphanumerically within
        groups that vary at the same rate.
        """
        raise NotImplementedError

    def round_floats(self, specs, fp_precision):
        _round = lambda v, fp: np.round(v, fp) if (type(v) in np.sctypes['float']) else round(v, fp)
        return (dict((k, _round(v, fp_precision) if (type(v) in float_types) else v)
                     for (k,v) in spec.items()) for spec in specs)

    def next(self):
        """
        Called to get a list of specifications: dictionaries with parameter name
        keys and string values.
        """
        raise StopIteration

    def update(self, data):
        """
        Called to update the state of the iterator when dynamic= True.
        Typically this methods is receiving metric values generated by the
        previous set of tasks in order to determine the next desired point in
        the parameter space. If the update fails or data is None, StopIteration
        should be raised.
        """
        raise NotImplementedError

    def schedule(self):
        """
        Specifies the expected number of specifications that will be returned on
        future iterations if dynamic=True. This is simply a list of integers
        specifying the number of argument sets to be returned on each subsequent
        call to next(). Return None if scheduling information cnanot be
        estimated.
        """
        raise NotImplementedError

    def copy(self):
        """
        Convenience method to avoid using the specifier without exhausting it.
        """
        return copy.copy(self)

    def _collect_by_key(self,specs):
        """
        Returns a dictionary like object with the lists of values collapsed by
        their respective key. Useful to find varying vs constant keys and to
        find how fast keys vary.
        """
        # Collects key, value tuples as list of lists then flatten using chain
        allkeys = itertools.chain.from_iterable([[(k, run[k])  for k in run] for run in specs])
        collection = defaultdict(list)
        for (k,v) in allkeys: collection[k].append(v)
        return collection

    def show(self):
        """
        Convenience method to inspect the available argument values in
        human-readable format. When dynamic, not all argument values may be
        available.
        """

        copied = self.copy()
        enumerated = [el for el in enumerate(copied)]
        for (group_ind, specs) in enumerated:
            if len(enumerated) > 1: print("Group %d" % group_ind)
            ordering = self.constant_keys() + self.varying_keys()
            # Ordered nicely by varying_keys definition.
            spec_lines = [ ', '.join(['%s=%s' % (k, s[k]) for k in ordering]) for s in specs]
            print('\n'.join([ '%d: %s' % (i,l) for (i,l) in enumerate(spec_lines)]))

        if self.dynamic:
            print('Remaining arguments not available for %s' % self.__class__.__name__)

    def __str__(self):
        """
        Argument specifiers are expected to have a succinct representation that
        is both human-readable but correctly functions as a proper object
        representation ie. repr.
        """
        return repr(self)

    def __add__(self, other):
        """
        Concatenates two argument specifiers. See StaticConcatenate and
        DynamicConcatenate documentation respectively.
        """
        assert not (self.dynamic and other.dynamic), 'Cannot concatenate two dynamic specifiers.'

        if self.dynamic or other.dynamic: return DynamicConcatenate(self,other)
        else:                             return StaticConcatenate(self,other)

    def __mul__(self, other):
        """
        Takes the cartesian product of two argument specifiers. See
        StaticCartesianProduct and DynamicCartesianProduct documentation.
        """
        assert not (self.dynamic and other.dynamic), \
            'Cannot take Cartesian product two dynamic specifiers.'

        if self.dynamic or other.dynamic: return DynamicCartesianProduct(self, other)
        else:                             return StaticCartesianProduct(self, other)

    def _cartesian_product(self, first_specs, second_specs):
        """
        Takes the Cartesian product of the specifications. Result will contain N
        specifications where N = len(first_specs) * len(second_specs) and keys
        are merged.  Example: [{'a':1},{'b':2}] * [{'c':3},{'d':4}] =
        [{'a':1,'c':3},{'a':1,'d':4},{'b':2,'c':3},{'b':2,'d':4}]
        """
        return  [ dict(zip(
                          list(s1.keys()) + list(s2.keys()),
                          list(s1.values()) + list(s2.values())
                      ))
                 for s1 in first_specs for s2 in second_specs ]

    def _args_kwargs(self, spec, args):
        """
        Separates out args from kwargs given a list of non-kwarg arguments. When
         args list is empty, kwargs alone are returned.
        """
        if args ==[]: return spec
        arg_list = [v for (k,v) in spec.items() if k in args]
        kwarg_dict = dict((k,v) for (k,v) in spec.items() if (k not in args))
        return (arg_list, kwarg_dict)

    def _setup_generator(self, args, review, log_file):
        """
        Basic setup before returning the arg-kwarg generator. Allows review and
        logging to be consistent with other useage.
        """
        all_keys =  self.constant_keys() + self.varying_keys()
        assert set(args) <= set(all_keys), 'Specified args must belong to the set of keys'

        if review:
            self.show()
            response =  None
            while response not in ['y','N', '']:
                response = raw_input('Continue? [y, N]: ')
            if response != 'y': return False

        if log_file is not None:
            try: log_file = open(os.path.abspath(log_file), 'w')
            except: raise Exception('Could not create log file at log_path')

        return log_file

    def _write_log(self, log_file, specs, line_no):
        """
        Writes out a log file to the root directory consistent Launcher useage.
        """
        lines = ['%d %s\n' % (line_no+i, json.dumps(spec)) for (i, spec) in enumerate(specs)]
        log_file.write('\n'.join(lines))
        return line_no + len(specs)

    def __call__(self, args=[], review=True, log_file=None):
        """
        A generator that allows argument specifiers to be used nicely with
        Python. For static specifiers, the args and kwargs are returned directly
        allowing iteration without unpacking. For dynamic specifiers, they are
        returned as lists for each concurrent group. By default all parameters
        are returned as kwargs but they can also returned as (args, kwargs) if
        the the arg keys are specified.
        """
        log_file = self._setup_generator(args, review, log_file)
        if log_file is False: return
        accumulator = []
        line_no = 0
        spec_iter = iter(self)
        while True:
            if self.dynamic:
                try:
                    specs = next(spec_iter)
                    retval = [self._args_kwargs(spec, args) for spec in specs]
                except: break
            if not self.dynamic and accumulator==[]:
                try:
                    accumulator = next(spec_iter)
                except: break
            if not self.dynamic:
                specs = accumulator[:1]
                retval = self._args_kwargs(specs[0], args)
                accumulator=accumulator[1:]

            if log_file is not None:
                line_no = self._write_log(log_file, specs, line_no)
            yield retval

        if log_file is not None: log_file.close()

class StaticArgs(BaseArgs):
    """
    Base class for many important static argument specifiers (dynamic=False)
    though also useful in its own right. Can be constructed from launcher log
    files and gives full control over the output arguments.Accepts the full
    static specification as a list of dictionaries, provides all necessary
    mechanisms for identifying varying and constant keys, implements next()
    appropriately, does not exhaust like dynamic specifiers and has useful
    support for len().
    """

    @staticmethod
    def extract_log_specification(log_path):
        """
        Parses the log file generated by a launcher and returns the corresponding
        tids and specification list. The specification list can be used to build a
        valid StaticArgs object. Converts keys from unicode to string for use as kwargs.

        e.g: s = StaticArgs(StaticArgs.extract_log_specification(<log_file>).values())
        """
        with open(log_path,'r') as log:
            splits = (line.split() for line in log)
            uzipped = ((int(split[0]), json.loads(" ".join(split[1:]))) for split in splits)
            szipped = [(i, dict((str(k),v) for (k,v) in d.items())) for (i,d) in uzipped]
        return dict(szipped)

    def __init__(self, specs, fp_precision=None, **kwargs):
        if fp_precision is None: fp_precision = BaseArgs.fp_precision
        self._specs = list(self.round_floats(specs, fp_precision))
        super(StaticArgs, self).__init__(dynamic=False, **kwargs)

    def __iter__(self):
        self._exhausted = False
        return self

    def next(self):
        if self._exhausted:
            raise StopIteration
        else:
            self._exhausted=True
            return self._specs

    def _unique(self, sequence, idfun=repr):
        """
        Note: repr() must be implemented properly on all objects. This is
        assumed by lancet when Python objects need to be formatted to string
        representation.
        """
        seen = {}
        return [seen.setdefault(idfun(e),e) for e in sequence
                if idfun(e) not in seen]

    def constant_keys(self):
        collection = self._collect_by_key(self._specs)
        return [k for k in collection if (len(self._unique(collection[k])) == 1)]

    def varying_keys(self):
        collection = self._collect_by_key(self._specs)
        constant_set = set(self.constant_keys())
        unordered_varying = set(collection.keys()).difference(constant_set)
        # Finding out how fast keys are varying
        grouplens = [(len([len(list(y)) for (_,y) in itertools.groupby(collection[k])]),k) for k in collection]
        varying_counts = [ (n,k) for (n,k) in sorted(grouplens) if (k in unordered_varying)]
        # Grouping keys with common frequency alphanumerically (desired behaviour).
        ddict = defaultdict(list)
        for (n,k) in varying_counts: ddict[n].append(k)
        alphagroups = [sorted(ddict[k]) for k in sorted(ddict)]
        return [el for group in alphagroups for el in group]

    def __len__(self): return len(self._specs)

    def __repr__(self):
        return "StaticArgs(%s)" % (self._specs)


class StaticConcatenate(StaticArgs):
    """
    StaticConcatenate is the sequential composition of two StaticArg
    specifiers. The specifier created by the compositon (firsts + second)
    generates the arguments in first followed by the arguments in
    second.
    """

    def __init__(self, first, second):

        self.first = first
        self.second = second
        specs = list(first.copy()(review=False)) + list(second.copy()(review=False))
        super(StaticConcatenate, self).__init__(specs)

    def __repr__(self):
        return "(%s + %s)" % (repr(self.first), repr(self.second))

class StaticCartesianProduct(StaticArgs):
    """
    StaticCartesianProduct is the cartesian product of two StaticArg
    specifiers. The specifier created by the compositon (firsts * second)
    generates the cartesian produce of the arguments in first followed by the
    arguments in second. Note that len(first * second) = len(first)*len(second)
    """
    def __init__(self, first, second):

        self.first = first
        self.second = second

        specs = self._cartesian_product(list(first.copy()(review=False)),
                                        list(second.copy()(review=False)))

        overlap = (set(self.first.varying_keys() + self.first.constant_keys())
                   &  set(self.second.varying_keys() + self.second.constant_keys()))
        assert overlap == set(), 'Sets of keys cannot overlap between argument specifiers in cartesian product.'
        super(StaticCartesianProduct, self).__init__(specs)

    def __repr__(self):   return '(%s * %s)' % (repr(self.first), repr(self.second))


class Args(StaticArgs):
    """
    Allows easy instantiation of a single set of arguments using keywords.
    Useful for instantiating constant arguments before applying a cartesian
    product.
    """

    def __init__(self, **kwargs):
        assert kwargs != {}, "Empty specification not allowed."
        fp_precision = kwargs.pop('fp_precision') if ('fp_precision' in kwargs) else BaseArgs.fp_precision
        specs = [dict((k, kwargs[k]) for k in kwargs)]
        super(Args,self).__init__(specs, fp_precision=fp_precision)

    def __repr__(self):
        spec = self._specs[0]
        return "Args(%s)"  % ', '.join(['%s=%s' % (k, fp_repr(v)) for (k,v) in spec.items()])


class LinearArgs(StaticArgs):
    """
    LinearArgs generates an argument that has a numerically interpolated range
    (linear by default). Values formatted to the given floating-point precision.
    """

    arg_name = param.String(default='arg_name',doc='''
         The name of the argument that is to be linearly varied over a numeric range.''')

    value =  param.Number(default=None, allow_None=True, constant=True, doc='''
         The starting numeric value of the linear interpolation.''')

    end_value = param.Number(default=None, allow_None=True, constant=True, doc='''
         The ending numeric value of the linear interpolation (inclusive).
         If not specified, will return 'value' (no interpolation).''')

    steps = param.Integer(default=1, constant=True, doc='''
         The number of steps to use in the interpolation. Default is 1.''')

    # Can't this be a lambda?
    mapfn = param.Callable(default=identityfn, constant=True, doc='''
         The function to be mapped across the linear range. Identity  by default ''')

    def __init__(self, arg_name, value, end_value=None,
                 steps=1, mapfn=identityfn, **kwargs):

        if end_value is not None:
            values = np.linspace(value, end_value, steps, endpoint=True)
            specs = [{arg_name:mapfn(val)} for val in values ]
        else:
            specs = [{arg_name:mapfn(value)}]
        self._pparams = ['end_value', 'steps', 'fp_precision', 'mapfn']

        super(LinearArgs, self).__init__(specs, arg_name=arg_name, value=value,
                                         end_value=end_value, steps=steps,
                                         mapfn=mapfn, **kwargs)

    def __repr__(self):
        modified = dict(self.get_param_values(onlychanged=True))
        pstr = ', '.join(['%s=%s' % (k, modified[k]) for k in self._pparams if k in modified])
        return "%s('%s', %s, %s)" % (self.__class__.__name__, self.arg_name, self.value, pstr)

class ListArgs(StaticArgs):
    """
    An argument specifier that takes its values from a given list.
    """

    arg_name = param.String(default='default',doc='''
         The key name that take its values from the given list.''')

    def __init__(self, arg_name, value_list, **kwargs):

        specs = [ {arg_name:val} for val in value_list]
        super(ListArgs, self).__init__(specs, arg_name=arg_name, **kwargs)

    def __repr__(self):
        value_list = (fp_repr(spec[self.arg_name]) for spec in self._specs)
        return "%s('%s',[%s])" % (self.__class__.__name__, self.arg_name, ', '.join(value_list))

#=============================#
# Dynamic argument specifiers #
#=============================#

class DynamicConcatenate(BaseArgs):
    def __init__(self, first, second):
        self.first = first
        self.second = second
        super(Concatenate, self).__init__(dynamic=True)

        self._exhausted = False
        self._first_sent = False
        self._first_cached = None
        self._second_cached = None
        if not first.dynamic: self._first_cached = next(first.copy())
        if not second.dynamic: self._second_cached = next(second.copy())

    def schedule(self):
        if self._first_cached is None:
            first_schedule = self.first.schedule()
            if first_schedule is None: return None
            return first_schedule + [len(self._second_cached)]
        else:
            second_schedule = self.second.schedule()
            if second_schedule is None: return None
            return [len(self._second_cached)]+ second_schedule

    def constant_keys(self):
        return list(set(self.first.constant_keys()) | set(self.second.constant_keys()))

    def varying_keys(self):
        return list(set(self.first.varying_keys()) | set(self.second.varying_keys()))

    def update(self, data):
        if (self.first.dynamic and not self._exhausted): self.first.update(data)
        elif (self.second.dynamic and self._first_sent): self.second.update(data)

    def next(self):
        if self._first_cached is None:
            try:  return next(self.first)
            except StopIteration:
                self._exhausted = True
                return self._second_cached
        else:
            if not self._first_sent:
                self._first_sent = True
                return self._first_cached
            else:
                return  next(self.second)

    def __repr__(self):
        return "(%s + %s)" % (repr(self.first), repr(self.second))

class DynamicCartesianProduct(BaseArgs):

    def __init__(self, first, second):

        self.first = first
        self.second = second

        overlap = set(self.first.varying_keys()) &  set(self.second.varying_keys())
        assert overlap == set(), 'Sets of keys cannot overlap between argument specifiers in cartesian product.'

        super(CartesianProduct, self).__init__(dynamic=True)

        self._first_cached = None
        self._second_cached = None
        if not first.dynamic: self._first_cached = next(first.copy())
        if not second.dynamic: self._second_cached = next(second.copy())

    def constant_keys(self):
        return list(set(self.first.constant_keys()) | set(self.second.constant_keys()))

    def varying_keys(self):
        return list(set(self.first.varying_keys()) | set(self.second.varying_keys()))

    def update(self, data):
        if self.first.dynamic:  self.first.update(data)
        if self.second.dynamic: self.second.update(data)

    def schedule(self):
        if self._first_cached is None:
            first_schedule = self.first.schedule()
            if first_schedule is None: return None
            return [len(self._second_cached)*i for i in first_schedule]
        else:
            second_schedule = self.second.schedule()
            if second_schedule is None: return None
            return [len(self._first_cached)*i for i in second_schedule]

    def next(self):
        if self._first_cached is None:
            first_spec = next(self.first)
            return self._cartesian_product(first_spec, self._second_cached)
        else:
            second_spec = next(self.second)
            return self._cartesian_product(self._first_cached, second_spec)

    def __repr__(self):   return '(%s * %s)' % (repr(self.first), repr(self.second))


#===================#
# Commands Template #
#===================#

class CommandTemplate(param.Parameterized):
    """
    A command template is a way of converting the key-value dictionary format
    returned by argument specifiers into a particular command. When called with
    an argument specifier, a command template returns a list of strings
    corresponding to a subprocess Popen argument list.

    __call__(self, spec, tid=None, info={}):

    All CommandTemplates must be callable. The tid argument is the task id and
    info is a dictionary of run-time information supplied by the launcher. Info
    contains of the following information : root_directory, timestamp,
    varying_keys, constant_keys, batch_name, batch_tag and batch_description.

    Command templates should avoid all platform specific logic (this is the job
    of the Launcher) but there are cases where the specification needs to be
    read from file. This allows tasks to be queued (typically on a cluster)
    before the required arguments are known. To achieve this two extra methods
    should be implemented if possible:

    specify(spec, tid, info):

    Takes a specification for the task and writes it to a file in the
    'specifications' subdirectory of the root directory. Each specification
    filename must include the tid to ensure uniqueness.

    queue(self,tid, info):

    The subprocess Popen argument list that upon execution runs the desired
    command using the arguments in the specification file of the given tid
    located in the 'specifications' folder of the root directory.
    """

    allowed_list = param.List(default=[], doc='''
        An optional, explicit list of the argument names that the
        CommandTemplate is expected to accept. If the empty list, no checking is
        performed. This allows some degree of error checking before tasks are
        launched. A command may exit if invalid parameters are supplied but it
        is often better to explicitly check: this avoids waiting in a cluster
        queue or invalid simulations due to unrecognised parameters.''')

    executable = param.String(default='python', constant=True, doc='''
        The executable that is to be run by this CommandTemplate. Unless the
        executable is a standard command expected on the system path, this
        should be an absolute path. By default this invokes python or the python
        environment used to invoke the CommandTemplate (eg. the topographica
        script).''')

    do_format = param.Boolean(default=True, doc= '''
        Set to True to receive input arguments as formatted strings, False for
        the raw unformatted objects.''')

    def __init__(self, executable=None, **kwargs):
        if executable is None:
            executable = sys.executable
        super(CommandTemplate,self).__init__(executable=executable, **kwargs)

    def __call__(self, spec, tid=None, info={}):
        """
        Formats a single argument specification - a dictionary of argument
        name/value pairs. The info dictionary includes the root_directory,
        batch_name, batch_tag, batch_description, timestamp, varying_keys,
        constant_keys.
        """
        raise NotImplementedError

    def _formatter(self, arg_specifier, spec):
        if self.do_format: return arg_specifier.spec_formatter(spec)
        else             : return spec

    def show(self, arg_specifier, file_handle=sys.stdout, queue_cmd_only=False):
        info = {'root_directory':     '<root_directory>',
                'batch_name':         '<batch_name>',
                'batch_tag':          '<batch_tag>',
                'batch_description':  '<batch_description>',
                'timestamp':          tuple(time.localtime()),
                'varying_keys':       arg_specifier.varying_keys(),
                'constant_keys':      arg_specifier.constant_keys()}

        if queue_cmd_only and not hasattr(self, 'queue'):
            print("Cannot show queue: CommandTemplate does not allow queueing")
            return
        elif queue_cmd_only:
            full_string = 'Queue command: '+ ' '.join([pipes.quote(el) for el in self.queue('<tid>',info)])
        elif (queue_cmd_only is False):
            copied = arg_specifier.copy()
            full_string = ''
            enumerated = list(enumerate(copied))
            for (group_ind, specs) in enumerated:
                if len(enumerated) > 1: full_string += "\nGroup %d" % group_ind
                quoted_cmds = [[pipes.quote(el) \
                                    for el in self(self._formatter(copied, s),'<tid>',info)] \
                              for s in specs]
                cmd_lines = [ '%d: %s\n' % (i, ' '.join(qcmds)) for (i,qcmds) in enumerate(quoted_cmds)]
                full_string += ''.join(cmd_lines)

        file_handle.write(full_string)
        file_handle.flush()

#===========#
# Launchers #
#===========#

class Launcher(param.Parameterized):
    """
    A Launcher is constructed using a name, an argument specifier and a command
    template and launches the corresponding tasks appropriately when invoked.

    This default Launcher uses subprocess to launch tasks. It is intended to
    illustrate the basic design and should be used as a base class for more
    complex Launchers. In particular all Launchers should retain the same
    behaviour of writing stdout/stderr to the streams directory, writing a log
    file and recording launch information.
    """

    arg_specifier = param.ClassSelector(BaseArgs, constant=True, doc='''
              The specifier used to generate the varying parameters for the tasks.''')

    command_template = param.ClassSelector(CommandTemplate, constant=True, doc='''
              The command template used to generate the commands for the current tasks.''')

    tag = param.String(default='', doc='''
               A very short, identifiable human-readable string that
               meaningfully describes the batch to be executed. Should not
               include spaces as it may be used in filenames.''')

    description = param.String(default='', doc='''
              A short description of the purpose of the current set of tasks.''')

    max_concurrency = param.Integer(default=2, allow_None=True, doc='''
             Concurrency limit to impose on the launch. As the current class
             uses subprocess locally, multiple processes are possible at
             once. Set to None for no limit (eg. for clusters)''')

    reduction_fn = param.Callable(default=None, doc='''
             A callable that will be invoked when the Launcher has completed all
             tasks. For example, this can be used to collect and analyse data
             generated across tasks (eg. reduce across maps in a
             RunBatchAnalysis), inform the user of completion (eg. send an
             e-mail) among other possibilities.''')

    timestamp = param.NumericTuple(default=(0,)*9, doc='''
            Optional override of timestamp (default timestamp set on launch
            call) in Python struct_time 9-tuple format.  Useful when you need to
            have a known root_directory path (see root_directory documentation)
            before launch. For example, you should store state related to
            analysis (eg. pickles) in the same location as everything else.''')

    timestamp_format = param.String(default='%Y-%m-%d_%H%M', allow_None=True, doc='''
             The timestamp format for the root directories in python datetime
             format. If None, the timestamp is omitted from root directory name.''')

    metric_loader = param.Callable(default=pickle.load, doc='''
             The function that will load the metric files generated by the
             task. By default uses pickle.load but json.load is also valid
             option. The callable must take a file object argument and return a
             python object. A third interchange format that might be considered
             is cvs but it is up to the user to implement the corresponding
             loader.''')

    @classmethod
    def resume_launch(cls):
        """
        Class method to allow Launchers to be controlled from the
        environment.  If the environment is processed and the launcher
        is resuming, return True, otherwise return False.
        """
        return False


    def __init__(self, batch_name, arg_specifier, command_template, **kwargs):

        super(Launcher,self).__init__(arg_specifier=arg_specifier,
                                          command_template = command_template,
                                          **kwargs)
        self.batch_name = batch_name
        self._spec_log = []
        if self.timestamp == (0,)*9:
            self.timestamp = tuple(time.localtime())

    def root_directory_name(self, timestamp=None):
        " A helper method that gives the root direcory name given a timestamp "
        if timestamp is None: timestamp = self.timestamp
        if self.timestamp_format is not None:
            return time.strftime(self.timestamp_format, timestamp) + '-' + self.batch_name
        else:
            return self.batch_name

    def append_log(self, specs):
        """
        The log contains the tids and corresponding specifications used during
        launch with the specifications in json format.
        """

        self._spec_log += specs
        with open(os.path.join(self.root_directory,
                               ("%s.log" % self.batch_name)), 'a') as log:
            lines = ['%d %s' % (tid, json.dumps(spec)) for (tid, spec) in specs]
            log.write('\n'.join(lines))

    def record_info(self):
        """
        All launchers should call this method to write the info file at the
        end of the launch.  The info file saves the full timestamp and launcher
        description. The file is written to the root_directory.
        """

        with open(os.path.join(self.root_directory,
            ('%s.info' % self.batch_name)), 'w') as info:
            startstr = time.strftime("Start Date: %d-%m-%Y Start Time: %H:%M (%Ss)",
                                     self.timestamp)
            endstr = time.strftime("Completed: %d-%m-%Y Start Time: %H:%M (%Ss)",
                                   time.localtime())
            lines = [startstr, endstr, 'Batch name: %s' % self.batch_name,
                     'Description: %s' % self.description]
            info.write('\n'.join(lines))

    def _setup_launch(self):
        """
        Method to be used by all launchers that prepares the root directory and
        generate basic launch information for command templates to use. Prepends
        some information to the description, registers a timestamp and return a
        dictionary of useful launch information constant across all tasks.
        """
        root_name = self.root_directory_name()
        self.root_directory = param.normalize_path(root_name)

        if not os.path.isdir(self.root_directory): os.makedirs(self.root_directory)
        metrics_dir = os.path.join(self.root_directory, 'metrics')
        if not os.path.isdir(metrics_dir) and self.arg_specifier.dynamic:
            os.makedirs(metrics_dir)

        formatstr = "Batch '%s' of tasks specified by %s and %s launched by %s with tag '%s':\n"
        classnames= [el.__class__.__name__ for el in
                [self.arg_specifier, self.command_template, self]]
        self.description = (formatstr % tuple([self.batch_name] + \
                classnames + [self.tag]))+self.description

        return {'root_directory':    self.root_directory,
                'timestamp':         self.timestamp,
                'varying_keys':      self.arg_specifier.varying_keys(),
                'constant_keys':     self.arg_specifier.constant_keys(),
                'batch_name':        self.batch_name,
                'batch_tag':         self.tag,
                'batch_description': self.description }

    def _setup_streams_path(self):
        streams_path = os.path.join(param.normalize_path(),
                               self.root_directory, "streams")

        try: os.makedirs(streams_path)
        except: pass
        # Waiting till these directories exist (otherwise potential qstat error)
        while not os.path.isdir(streams_path): pass
        return streams_path

    def extract_metrics(self, tids, launchinfo):
        """
        Method to extract the metrics generated by the tasks required to update
        the argument specifier (if dynamic). Uses the metric loader to extract
        the metric files of the 'metrics' subdirectory in the root
        directory. Metric files should always include the tid in their name for
        uniqueness and all metric files of the same tid will be returned
        together.
        """
        metrics_dir = os.path.join(self.root_directory, 'metrics')
        listing = os.listdir(metrics_dir)
        try:
            matches = [l for l in listing for tid in tids
                    if fnmatch.fnmatch(l, 'metric-%d-*' % tid)]
            pfiles = [open(os.path.join(metrics_dir, match),'rb') for match in matches]
            return [self.metric_loader(pfile) for pfile in pfiles]
        except:
            logging.error("Cannot load required metric files. Cannot continue.")
            return None # StopIteration should be raised by the argument specifier

    def limit_concurrency(self, elements):
        """
        Helper function that breaks list of elements into chunks (sublists) of
        size self.max_concurrency.
        """
        if self.max_concurrency is None: return [elements]

        return [elements[i:i+self.max_concurrency] for i in
               range(0, len(elements), self.max_concurrency)]

    def launch(self):
        """
        The method that starts Launcher execution. Typically called by a launch
        helper.  This could be called directly by the users but the risk is that
        if __name__=='__main__' is omitted, the launcher may rerun on any import
        of the script effectively creating a fork-bomb.
        """
        launchinfo = self._setup_launch()
        streams_path = self._setup_streams_path()

        last_tid = 0
        last_tids = []
        for gid, groupspecs in enumerate(self.arg_specifier):
            tids = list(range(last_tid, last_tid+len(groupspecs)))
            last_tid += len(groupspecs)
            allcommands = [self.command_template(
                                self.command_template._formatter(self.arg_specifier, spec), tid, launchinfo) \
                           for (spec,tid) in zip(groupspecs,tids)]

            self.append_log(list(zip(tids,groupspecs)))
            batches = self.limit_concurrency(list(zip(allcommands,tids)))
            for bid, batch in enumerate(batches):
                processes = []
                stdout_handles = []
                stderr_handles = []
                for (cmd,tid) in batch:
                    stdout_handle = open(os.path.join(streams_path, "%s.o.%d" % (self.batch_name, tid)), "wb")
                    stderr_handle = open(os.path.join(streams_path, "%s.e.%d" % (self.batch_name, tid)), "wb")
                    processes.append(subprocess.Popen(cmd, stdout=stdout_handle, stderr=stderr_handle))
                    stdout_handles.append(stdout_handle)
                    stderr_handles.append(stderr_handle)

                logging.info("Batch of %d (%d:%d/%d) subprocesses started..." % \
                            (len(processes), gid, bid, len(batches)-1))

                for p in processes: p.wait()

                for stdout_handle in stdout_handles: stdout_handle.close()
                for stderr_handle in stderr_handles: stderr_handle.close()

            last_tids = tids[:]

            if self.arg_specifier.dynamic:
                self.arg_specifier.update(self.extract_metrics(last_tids, launchinfo))

        self.record_info()
        if self.reduction_fn is not None: self.reduction_fn(self._spec_log, self.root_directory)

class QLauncher(Launcher):
    """
    Launcher that operates the Sun Grid Engine using default arguments suitable
    for running on the Edinburgh Eddie cluster. Allows automatic parameter
    search strategies such as hillclimbing to be used, queueing jobs without
    arguments without blocking (via specification files).

    One of the main features of this class is that it is non-blocking - it alway
    exits shortly after invoking qsub. This means that the script is not left
    running, waiting for long periods of time on the cluster. This is
    particularly important for long simulation where you wish to run some code
    at the end of the simulation (eg. plotting your results) or when waiting for
    results from runs (eg. waiting for results from 100 seeds to update your
    hillclimbing algorithm).

    To achieve this, QLauncher qsubs a job that relaunches the user's lancet
    script which can then instructs the Qlauncher to continue with
    'collate_and_launch' step (via environment variable). Collating refers to
    either collecting output from a subset of runs to update the argument specifier
    or to a final reduction operation over all the results. Jobs are qsubbed
    with dependencies on the previous collate step and conversely each collate
    step depends on the all the necessary tasks steps reaching completion.

    By convention the standard output and error streams go to the corresponding
    folders in the 'streams' subfolder of the root directory - any -o or -e qsub
    options will be overridden. The job name (the -N flag) is specified
    automatically and any user value will be ignored.
    """

    qsub_switches = param.List(default=['-V', '-cwd'], doc = '''
          Specifies the qsub switches (flags without arguments) as a list of
          strings. By default the -V switch is used to exports all environment
          variables in the host environment to the batch job.''')

    qsub_flag_options = param.Dict(default={'-b':'y'}, doc='''
          Specifies qsub flags and their corresponding options as a
          dictionary. Valid values may be strings or lists of string.  If a
          plain Python dictionary is used, the keys are alphanumerically sorted,
          otherwise the dictionary is assumed to be an OrderedDict (Python 2.7+,
          Python3 or param.external.OrderedDict) and the key ordering will be
          preserved.

          By default the -b (binary) flag is set to 'y' to allow binaries to be
          directly invoked. Note that the '-' is added to the key if missing (to
          make into a valid flag) so you can specify using keywords in the dict
          constructor: ie. using qsub_flag_options=dict(key1=value1,
          key2=value2, ....)''')

    script_path = param.String(default=None, allow_None = True, doc='''
         For python environments, this is the path to the lancet script
         allowing the QLauncher to collate jobs. The lancet script is run with
         the LANCET_ANALYSIS_DIR environment variable set appropriately. This
         allows the launcher to resume launching jobs when using dynamic
         argument specifiers or when performing a reduction step.

         If set to None, the command template executable (whatever it may be) is
         executed with the environment variable set.''')

    @classmethod
    def resume_launch(cls):
        """
        Resumes the execution of the launcher if environment contains
        LANCET_ANALYSIS_DIR. This information allows the
        launcher.pickle file to be unpickled to resume the launch.
        """
        if "LANCET_ANALYSIS_DIR" not in os.environ: return False

        root_path = param.normalize_path(os.environ["LANCET_ANALYSIS_DIR"])
        del os.environ["LANCET_ANALYSIS_DIR"]
        pickle_path = os.path.join(root_path, 'launcher.pickle')
        launcher = pickle.load(open(pickle_path,'rb'))
        launcher.collate_and_launch()

        return True

    def __init__(self, batch_name, arg_specifier, command_template, **kwargs):
        assert "LANCET_ANALYSIS_DIR" not in os.environ, "LANCET_ANALYSIS_DIR already in environment!"

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
        self.is_dynamic_qsub = all([self.arg_specifier.dynamic,
                                    hasattr(self.arg_specifier, 'schedule'),
                                    hasattr(self.command_template,   'queue'),
                                    hasattr(self.command_template,   'specify')])

    def qsub_args(self, override_options, cmd_args, append_options=[]):
        """
        Method to generate Popen style argument list for qsub using the
        qsub_switches and qsub_flag_options parameters. Switches are returned
        first. The qsub_flag_options follow in keys() ordered if not a vanilla
        Python dictionary (ie. a Python 2.7+ or param.external OrderedDict).
        Otherwise the keys are sorted alphanumerically. Note that
        override_options is a list of key-value pairs.
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
        Main entry point for the launcher. Collects the static information about
        the launch and sets up the stdout and stderr stream output
        directories. Generates the first call to collate_and_launch().
        """
        self._launchinfo = self._setup_launch()
        self.job_timestamp = time.strftime('%H%M%S')

        streams_path = self._setup_streams_path()

        self.qsub_flag_options['-o'] = streams_path
        self.qsub_flag_options['-e'] = streams_path

        self.collate_and_launch()

    def collate_and_launch(self):
        """
        Method that collates the previous jobs and launches the next block of
        concurrent jobs. The launch type can be either static or dynamic (using
        schedule, queue and specify for dynamic argument specifiers).  This method is
        invoked on initial launch and then subsequently via the commandline to
        collate the previously run jobs and launching the next block of jobs.
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
        if self.arg_specifier.dynamic:
            self.arg_specifier.update(self.extract_metrics(self.last_tids, self._launchinfo))
        self.last_tids = [tid for (tid,_) in tid_specs]

        output_dir = self.qsub_flag_options['-o']
        error_dir = self.qsub_flag_options['-e']
        if self.is_dynamic_qsub: self.dynamic_qsub(output_dir, error_dir, tid_specs)
        else:            self.static_qsub(output_dir, error_dir, tid_specs)

        # Pickle launcher before exit if necessary.
        if (self.arg_specifier.dynamic) or (self.reduction_fn is not None):
            root_path = param.normalize_path(self.root_directory)
            pickle_path = os.path.join(root_path, 'launcher.pickle')
            pickle.dump(self, open(pickle_path,'wb'))

    def qsub_collate_and_launch(self, output_dir, error_dir, job_names):
        """
        The method that actually runs qsub to invoke the user launch script with
        the necessary environment variable to trigger the next collation step
        and next block of jobs.
        """

        job_name = "%s_%s_collate_%d" % (self.batch_name,
                                         self.job_timestamp,
                                         self.collate_count)

        overrides = [("-e",error_dir), ('-N',job_name), ("-o",output_dir),
                     ('-hold_jid',','.join(job_names))]

        cmd_args = [self.command_template.executable]
        if self.script_path is not None: cmd_args += [self.script_path]

        popen_args = self.qsub_args(overrides, cmd_args,
                        [("-v", "LANCET_ANALYSIS_DIR=%s" % self.root_directory)])

        p = subprocess.Popen(popen_args, stdout=subprocess.PIPE)
        (stdout, stderr) = p.communicate()

        self.collate_count += 1
        logging.debug(stdout)
        logging.info("Invoked qsub for next batch.")
        return job_name

    def static_qsub(self, output_dir, error_dir, tid_specs):
        """
        This method handles static argument specifiers and cases where the
        dynamic specifiers cannot be queued before the arguments are known.
        """
        processes = []
        job_names = []

        for (tid, spec) in tid_specs:
            job_name = "%s_%s_job_%d" % (self.batch_name, self.job_timestamp, tid)
            job_names.append(job_name)
            cmd_args = self.command_template(
                    self.command_template._formatter(self.arg_specifier, spec),
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
        This method handles dynamic argument specifiers where the dynamic
        argument specifier can be queued before the arguments are computed.
        """

        # Write out the specification files in anticipation of execution
        for (tid, spec) in tid_specs:
            self.command_template.specify(
                    self.command_template._formatter(self.arg_specifier, spec),
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
        Runs qdel command to remove all remaining queued jobs using the
        <batch_name>* pattern . Necessary when StopIteration is raised with
        scheduled jobs left on the queue.
        """
        p = subprocess.Popen(['qdel', '%s_%s*' % (self.batch_name, self.job_timestamp)],
                             stdout=subprocess.PIPE)
        (stdout, stderr) = p.communicate()


#===============#
# Launch Helper #
#===============#

class review_and_launch(param.Parameterized):
    """
    The basic example of the sort of helper that is highly recommended for
    launching:

    1) The lancet script may include objects/class that need to be imported
    (eg. for accessing analysis functions) to execute the tasks. By default this
    would execute the whole script and therefore re-run the Launcher which would
    cause a fork-bomb! This decorator only executes the launcher that is
    returned by the wrapped function if __name__=='__main__'

    Code in the script after Launcher execution has been invoked is not
    guaranteed to execute *after* all tasks are complete (eg. due to forking,
    subprocess, qsub etc). This decorator helps solves this issue, making sure
    launch is the last thing in the definition function. The reduction_fn
    parameter is the proper way of executing code after the Launcher exits."""

    launcher_class = param.Parameter(doc='''
         The launcher class used for this lancet script.  Necessary to access
         launcher classmethods (to resume launch for example).''')

    output_directory = param.String(default='.', doc='''
         The output directory - the directory that will contain all the root
         directories for the individual launches.''')

    review = param.Boolean(default=True, doc='''
         Whether or not to perform a detailed review of the launch.''')

    main_script = param.Boolean(default=True, doc='''
         Whether the launch is occuring from a lancet script running as
         main. Set to False for projects using lancet outside the main script.''')

    launch_args = param.ClassSelector(default=None, allow_None=True, class_=StaticArgs,
         doc= '''An optional argument specifier to parameterise lancet,
                 allowing multi-launch scripts.  Useful for collecting
                 statistics over runs that are not deterministic or are affected
                 by a random seed for example.''')

    def __init__(self, launcher_class, output_directory='.', **kwargs):

        super(review_and_launch, self).__init__(launcher_class=launcher_class,
                                                output_directory=output_directory,
                                                **kwargs)
        self._get_launcher = lambda x:x
        self._cross_checks = [(self._get_launcher, self.cross_check_launchers)]
        self._reviewers = [(self._get_launcher, self.review_launcher ),
                           (self._get_launcher, self.review_args),
                           (self._get_launcher, self.review_command_template)]

    def configure_launch(self, lval):
        """
        Hook to allow the Launch helper to autoconfigure launches as
        appropriate.  For example, can be used to save objects in their final
        state. Return True if configuration successful, False to cancel the
        entire launch.
        """
        return True

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
        batch_names = [launcher.batch_name for launcher in launchers]
        timestamps = [launcher.timestamp for launcher in launchers]
        launcher_classes = [launcher.__class__ for launcher in launchers]

        if len(set(batch_names)) != len(launchers):
            raise Exception('Each launcher requires a unique batch name.')

        if not all(timestamps[0] == tstamp for tstamp in timestamps):
            raise Exception("Launcher timestamps not all equal. Consider setting timestamp explicitly.")

        if not all(lclass == self.launcher_class for lclass in launcher_classes):
            raise Exception("Launcher class inconsistent with returned launcher instances.")

        # Argument name consistency checks
        checkable_launchers = [launcher for launcher in launchers
                               if (launcher.command_template.allowed_list != [])]

        used_args = [set(launcher.arg_specifier.varying_keys()
                         + launcher.arg_specifier.constant_keys()) for launcher in checkable_launchers]

        allowed_args = [set(launcher.command_template.allowed_list) for launcher in checkable_launchers]

        clashes = [used - allowed for (used, allowed) in zip(used_args, allowed_args)
                   if (used - allowed) != set()]

        if clashes != []: raise Exception("Keys %s not in CommandTemplate allowed list" % list(clashes[0]))

    def __call__(self, f):
        if self.main_script and f.__module__ != '__main__': return None

        # Resuming launch as necessary
        if self.launcher_class.resume_launch(): return None

        # Setting the output directory via param
        if self.output_directory is not None:
            param.normalize_path.prefix = self.output_directory

        # Calling the wrapped function with appropriate arguments
        kwargs_list = [{}] if (self.launch_args is None) else list(self.launch_args(review=False))
        lvals = [f(**kwargs_list[0])]
        if self.launch_args is not None:
            self.launcher_class.timestamp = self._get_launcher(lvals[0]).timestamp
            lvals += [f(**kwargs) for kwargs in kwargs_list[1:]]

        # Cross checks
        for (accessor, checker) in self._cross_checks:
            checker([accessor(lval) for lval in lvals])

        if self.review:
            # Run review of launch args only if necessary
            if self.launch_args is not None:
                proceed = self.review_args(self.launch_args, heading='Lancet Meta Parameters')
                if not proceed: return None

            for (count, lval) in enumerate(lvals):
                proceed = all(r(access(lval)) for (access, r) in self._reviewers)
                if not proceed:
                    print("Aborting launch.")
                    return None

                if len(lvals)!= 1 and count < len(lvals)-1:
                    skip_remaining = self.input_options(['Y', 'n','quit'],
                                     '\nSkip remaining reviews?', default='y')
                    if skip_remaining == 'quit': return None
                    if skip_remaining == 'y': break

        response = self.input_options(['y','N'], 'Execute?', default='n')
        if response == 'y':
            # Run configure hook. Exit if any configuration fails
            if not all([self.configure_launch(lval) for lval in lvals]): return

            for lval in lvals:
                launcher =  self._get_launcher(lval)
                print("== Launching  %s ==" % launcher.batch_name)
                launcher.launch()

    def review_launcher(self, launcher):
        command_template = launcher.command_template
        print(self.section('Launcher'))
        print("Type: %s" % launcher.__class__.__name__)
        print("Batch Name: %s" % launcher.batch_name)
        print("Command executable: %s" % command_template.executable)
        root_directory = launcher.root_directory_name()
        print("Root directory: %s" % param.normalize_path(root_directory))
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
              (arg_specifier.__class__.__name__, arg_specifier.dynamic))
        print("Varying Keys: %s" % arg_specifier.varying_keys())
        print("Constant Keys: %s" % arg_specifier.constant_keys())
        print("Definition: %s" % arg_specifier)

        response = self.input_options(['Y', 'n','quit'],
                '\nShow available argument specifier entries?', default='y')
        if response == 'quit': return False
        if response == 'y':  arg_specifier.show()
        print
        return True

    def review_command_template(self, launcher):
        command_template = launcher.command_template
        arg_specifier = launcher.arg_specifier
        print(self.section(command_template.__class__.__name__))
        if command_template.allowed_list != []:
            print("Allowed List: %s" % command_template.allowed_list)
        if isinstance(launcher, QLauncher) and launcher.is_dynamic_qsub:
            command_template.show(arg_specifier, queue_cmd_only=True)

        response = self.input_options(['Y', 'n','quit','save'],
                                      '\nShow available command entries?', default='y')
        if response == 'quit': return False
        if response == 'y':
            command_template.show(arg_specifier)
        if response == 'save':
            fname = raw_input('Filename: ').replace(' ','_')
            with open(os.path.abspath(fname),'w') as f:
                command_template.show(arg_specifier, file_handle=f)
        print
        return True
