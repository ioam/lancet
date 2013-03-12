#
# Lancet core
#

import os, sys, time, pipes, subprocess, itertools, copy
import re, glob, fnmatch, string, re
import json, pickle
import logging

import param
try:
    import numpy as np
    np_ftypes = np.sctypes['float']
except:
    np, np_ftypes = None, []

from collections import defaultdict

# Points to lancet.core or lancet.ipython as appropriate
# Necessary for functions that refer to classes with alternative
# implementations provided by lancet.ipython
import lancet
lancet._module = sys.modules[__name__]

float_types = [float] + np_ftypes
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

    fp_precision = param.Integer(default=4, constant=True, doc='''
         The floating point precision to use for floating point values.  Unlike
         other basic Python types, floats need care with their representation as
         you only want to display up to the precision actually specified.''')

    def __init__(self, **params):
        super(BaseArgs,self).__init__(**params)
        self._pprint_args = ([],[],None,{})
        self.pprint_args([],['fp_precision', 'dynamic'])

    def __iter__(self): return self

    def __contains__(self, value):
        return value in (self.constant_keys() + self.constant_keys())

    def spec_formatter(self, spec):
        " Formats the elements of an argument set appropriately"
        return dict((k, str(v)) for (k,v) in spec.items())

    def constant_keys(self):
        """
        Returns the list of parameter names whose values are constant as the
        argument specifier is iterated.  Note that the union of constant and
        varying_keys should partition the entire set of keys.
        """
        raise NotImplementedError

    def constant_items(self):
        """
        Returns the set of constant items as a list of tuples. This allows easy
        conversion to dictionary format. Note, the items should be supplied in
        the same key ordering as for constant_keys() for consistency.
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
        _round = lambda v, fp: np.round(v, fp) if (type(v) in np_ftypes) else round(v, fp)
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
            spec_lines = [', '.join(['%s=%s' % (k, s[k]) for k in ordering]) for s in specs]
            print('\n'.join(['%d: %s' % (i,l) for (i,l) in enumerate(spec_lines)]))

        if self.dynamic:
            print('Remaining arguments not available for %s' % self.__class__.__name__)

    def __add__(self, other):
        """
        Concatenates two argument specifiers. See StaticConcatenate and
        DynamicConcatenate documentation respectively.
        """
        if not other: return self
        assert not (self.dynamic and other.dynamic), 'Cannot concatenate two dynamic specifiers.'

        if self.dynamic or other.dynamic: return lancet._module.DynamicConcatenate(self,other)
        else:                             return lancet._module.StaticConcatenate(self,other)

    def __mul__(self, other):
        """
        Takes the cartesian product of two argument specifiers. See
        StaticCartesianProduct and DynamicCartesianProduct documentation.
        """
        if not other: return []
        assert not (self.dynamic and other.dynamic), \
            'Cannot take Cartesian product two dynamic specifiers.'

        if self.dynamic or other.dynamic: return lancet._module.DynamicCartesianProduct(self, other)
        else:                             return lancet._module.StaticCartesianProduct(self, other)

    def __radd__(self, other):
        if not other: return self

    def __rmul__(self, other):
        if not other: return []

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

    def pprint_args(self, pos_args, keyword_args, infix_operator=None, extra_params={}):
        """
        Method to define the positional arguments and keyword order for pretty printing.
        """
        if infix_operator and not (len(pos_args)==2 and keyword_args==[]):
            raise Exception('Infix format requires exactly two positional arguments and no keywords')
        (kwargs,_,_,_) = self._pprint_args
        self._pprint_args = (keyword_args + kwargs, pos_args, infix_operator, extra_params)

    def _pprint(self, cycle=False, flat=False, annotate=False, level=1, tab = '   '):
        """
        Pretty printer that prints only the modified keywords and generates flat
        representations (for repr) and optionally annotates with a comment.
        """
        (kwargs, pos_args, infix_operator, extra_params) = self._pprint_args
        (br, indent)  = ('' if flat else '\n', '' if flat else tab * level)
        prettify = lambda x: isinstance(x, BaseArgs) and not flat
        pretty = lambda x: x._pprint(flat=flat, level=level+1) if prettify(x) else repr(x)

        params = dict(self.get_param_values())
        modified = [k for (k,v) in self.get_param_values(onlychanged=True)]
        pkwargs = [(k, params[k])  for k in kwargs if (k in modified)] + extra_params.items()
        arg_list = [(k,params[k]) for k in pos_args] + pkwargs

        len_ckeys, len_vkeys = len(self.constant_keys()), len(self.varying_keys())
        info_triple = (len(self),
                       ', %d constant key(s)' % len_ckeys if len_ckeys else '',
                       ', %d varying key(s)'  % len_vkeys if len_vkeys else '')
        annotation = '# == %d items%s%s ==\n' % info_triple
        lines = [annotation] if annotate else []    # Optional annotating comment

        if cycle:
            lines.append('%s(...)' % self.__class__.__name__)
        elif infix_operator:
            level = level - 1
            triple = (pretty(params[pos_args[0]]), infix_operator, pretty(params[pos_args[1]]))
            lines.append('%s %s %s' % triple)
        else:
            lines.append('%s(' % self.__class__.__name__)
            for (k,v) in arg_list:
                lines.append('%s%s=%s' %  (br+indent, k, pretty(v)))
                lines.append(',')
            lines = lines[:-1] +[br+(tab*(level-1))+')'] # Remove trailing comma

        return ''.join(lines)

    def __repr__(self):  return self._pprint(flat=True)
    def __str__(self): return self._pprint()

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

    specs = param.List(default=[], constant=True, doc='''
          The static list of specifications (ie. dictionaries) to be
          returned by the specifier. Float values are rounded to
          fp_precision.''')

    def __init__(self, specs, fp_precision=None, **kwargs):
        if fp_precision is None: fp_precision = BaseArgs.fp_precision
        specs = list(self.round_floats(specs, fp_precision))
        super(StaticArgs, self).__init__(dynamic=False, fp_precision=fp_precision, specs=specs, **kwargs)
        self.pprint_args(['specs'],[])

    def __iter__(self):
        self._exhausted = False
        return self

    def next(self):
        if self._exhausted:
            raise StopIteration
        else:
            self._exhausted=True
            return self.specs

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
        collection = self._collect_by_key(self.specs)
        return [k for k in sorted(collection) if (len(self._unique(collection[k])) == 1)]

    def constant_items(self):
        collection = self._collect_by_key(self.specs)
        return [(k,collection[k][0]) for k in self.constant_keys()]

    def varying_keys(self):
        collection = self._collect_by_key(self.specs)
        constant_set = set(self.constant_keys())
        unordered_varying = set(collection.keys()).difference(constant_set)
        # Finding out how fast keys are varying
        grouplens = [(len([len(list(y)) for (_,y) in itertools.groupby(collection[k])]),k) for k in collection]
        varying_counts = [(n,k) for (n,k) in sorted(grouplens) if (k in unordered_varying)]
        # Grouping keys with common frequency alphanumerically (desired behaviour).
        ddict = defaultdict(list)
        for (n,k) in varying_counts: ddict[n].append(k)
        alphagroups = [sorted(ddict[k]) for k in sorted(ddict)]
        return [el for group in alphagroups for el in group]

    def __len__(self): return len(self.specs)

class StaticConcatenate(StaticArgs):
    """
    StaticConcatenate is the sequential composition of two StaticArg
    specifiers. The specifier created by the compositon (firsts + second)
    generates the arguments in first followed by the arguments in
    second.
    """

    first = param.ClassSelector(default=None, class_=StaticArgs, allow_None=True, constant=True, doc='''
            The first static specifier used to generate the concatenation.''')

    second = param.ClassSelector(default=None, class_=StaticArgs, allow_None=True, constant=True, doc='''
            The second static specifier used to generate the concatenation.''')

    def __init__(self, first, second):

        max_precision = max(first.fp_precision, second.fp_precision)
        specs = first.specs + second.specs
        super(StaticConcatenate, self).__init__(specs, fp_precision=max_precision,
                                                first=first, second=second)
        self.pprint_args(['first', 'second'],[], infix_operator='+')

class StaticCartesianProduct(StaticArgs):
    """
    StaticCartesianProduct is the cartesian product of two StaticArg
    specifiers. The specifier created by the compositon (firsts * second)
    generates the cartesian produce of the arguments in first followed by the
    arguments in second. Note that len(first * second) = len(first)*len(second)
    """

    first = param.ClassSelector(default=None, class_=StaticArgs, allow_None=True, constant=True, doc='''
            The first static specifier used to generate the Cartesian product.''')

    second = param.ClassSelector(default=None, class_=StaticArgs, allow_None=True, constant=True, doc='''
            The second static specifier used to generate the Cartesian product.''')

    def __init__(self, first, second):

        max_precision = max(first.fp_precision, second.fp_precision)
        specs = self._cartesian_product(first.specs, second.specs)

        overlap = (set(first.varying_keys() + first.constant_keys())
                   &  set(second.varying_keys() + second.constant_keys()))
        assert overlap == set(), 'Sets of keys cannot overlap between argument specifiers in cartesian product.'
        super(StaticCartesianProduct, self).__init__(specs, fp_precision=max_precision,
                                                     first=first, second=second)
        self.pprint_args(['first', 'second'],[], infix_operator='*')

class Args(StaticArgs):
    """
    Allows easy instantiation of a single set of arguments using keywords.
    Useful for instantiating constant arguments before applying a cartesian
    product.
    """

    def __init__(self, **kwargs):
        assert kwargs != {}, "Empty specification not allowed."
        fp_precision = kwargs.pop('fp_precision') if ('fp_precision' in kwargs) else None
        specs = [dict((k, kwargs[k]) for k in kwargs)]
        super(Args,self).__init__(specs, fp_precision=fp_precision)
        self.pprint_args([],list(kwargs.keys()), None, dict(**kwargs))

class LinearArgs(StaticArgs):
    """
    LinearArgs generates an argument from a numerically interpolated range which
    is linear by default. An optional function can be specified to sample a
    numeric range with regular intervals.
    """

    key = param.String(default='', doc='''
         The key assigned to values computed over the linear numeric range.''')

    start_value =  param.Number(default=None, allow_None=True, constant=True, doc='''
         The starting numeric value of the linear interpolation.''')

    end_value = param.Number(default=None, allow_None=True, constant=True, doc='''
         The ending numeric value of the linear interpolation (inclusive).''')

    steps = param.Integer(default=2, constant=True, bounds=(2,None), doc='''
         The number of steps to interpolate over. Default is 2 which returns the
         start and end values without interpolation.''')

    # Can't this be a lambda?
    mapfn = param.Callable(default=identityfn, constant=True, doc='''
         The function to be mapped across the linear range. Identity  by default ''')

    def __init__(self, key, start_value, end_value, steps=2, mapfn=identityfn, **kwargs):

        values = self.linspace(start_value, end_value, steps)
        specs = [{key:mapfn(val)} for val in values ]

        super(LinearArgs, self).__init__(specs, key=key, start_value=start_value,
                                         end_value=end_value, steps=steps,
                                         mapfn=mapfn, **kwargs)
        self.pprint_args(['key', 'start_value'], ['end_value', 'steps'])

    def linspace(self, start, stop, n):
        """ Nice simple replacement for numpy linspace"""
        L = [0.0] * n
        nm1 = n - 1
        nm1inv = 1.0 / nm1
        for i in range(n):
            L[i] = nm1inv * (start*(nm1 - i) + stop*i)
        return L

class ListArgs(StaticArgs):
    """
    An argument specifier that takes its values from a given list.
    """

    list_values = param.List(default=[], constant=True, doc='''
         The list values that are to be returned by the specifier''')

    list_key = param.String(default='default', constant=True, doc='''
         The key assigned to the elements of the given list.''')

    def __init__(self, list_key, list_values, **kwargs):

        assert list_values != [], "Empty list not allowed."
        specs = [{list_key:val} for val in list_values]
        super(ListArgs, self).__init__(specs, list_key=list_key, list_values=list_values, **kwargs)
        self.pprint_args(['list_key', 'list_values'], [])

class Log(StaticArgs):
    """
    Specifier that loads arguments from a log file in tid (task id) order.  For
    full control over the arguments, you can use this class with StaticArgs as
    follows: StaticArgs(Log.extract_log(<log_file>).values()),

    This wrapper class allows a concise representation of log specifiers with
    the option of adding the task id to the loaded specifications.
    """

    log_path = param.String(default=None, allow_None=True, constant=True, doc='''
              The relative or absolute path to the log file. If a relative path
              is given, the absolute path is computed with param.normalize_path
              (os.getcwd() by default).''')

    tid_key = param.String(default='tid', constant=True, allow_None=True, doc='''
               If not None, the key given to the tid values included in the
               loaded specifications. If None, the tid number is ignored.''')

    @staticmethod
    def extract_log(log_path, dict_type=dict):
        """
        Parses the log file generated by a launcher and returns dictionary with
        tid keys and specification values.

        Ordering can be maintained by setting dict_type to the appropriate
        constructor. Keys are converted from unicode to strings for kwarg use.
        """
        with open(param.normalize_path(log_path),'r') as log:
            splits = (line.split() for line in log)
            uzipped = ((int(split[0]), json.loads(" ".join(split[1:]))) for split in splits)
            szipped = [(i, dict((str(k),v) for (k,v) in d.items())) for (i,d) in uzipped]
        return dict_type(szipped)

    @staticmethod
    def write_log(log_path, data, allow_append=True):
        """
        Writes the supplied specifications to the log path. The data may be
        supplied as either as a StaticSpecifier or as a list of dictionaries.

        By default, specifications will be appropriately appended to an existing
        log file. This can be disabled by setting allow_append to False.
        """
        append = os.path.isfile(log_path)
        listing = isinstance(data, list)

        if append and not allow_append:
            raise Exception('Appending has been disabled and file %s exists' % log_path)

        if not (listing or isinstance(data, StaticArgs)):
            raise Exception('Can only write static specifiers or dictionary lists to log file.')

        specs = data if listing else data.specs
        if not all(isinstance(el,dict) for el in specs):
            raise Exception('List elements must be dictionaries.')

        log_file = open(log_path, 'r+') if append else open(log_path, 'w')
        start = int(log_file.readlines()[-1].split()[0])+1 if append else 0
        ascending_indices = range(start, start+len(data))

        log_str = '\n'.join(['%d %s' % (tid, json.dumps(el)) for (tid, el) in zip(ascending_indices, specs)])
        log_file.write("\n"+log_str if append else log_str)
        log_file.close()

    def __init__(self, log_path, tid_key='tid', **kwargs):

        log_items = sorted(Log.extract_log(log_path).iteritems())

        if tid_key is not None:
            log_specs = [dict(spec.items()+[(tid_key,idx)]) for (idx, spec) in log_items]
        else:
            log_specs = [spec for (_, spec) in log_items]
        super(Log, self).__init__(log_specs, log_path=log_path, tid_key=tid_key, **kwargs)
        self.pprint_args(['log_path'], ['tid_key'])

class Indexed(StaticArgs):
    """
    Given two StaticArgs, link the arguments of via an index value.
    The index value of the given key must have a matching entry in the
    index. Once a match is found, the results are merged with the
    resulting ordering identical to that of the input operand.

    The value used for matching is specified by the key
    name. Uniqueness of keys in the index is enforced and these keys
    must be a superset of those expressed by the operand.

    By default, fp_precision is the maximum of that used by the
    operand and index.
    """
    operand = param.ClassSelector(default=None, class_=StaticArgs, allow_None=True, constant=True, doc='''
              The source specifier from which the index_key is extracted for
              looking up the corresponding value in the index.''')

    index = param.ClassSelector(default=None, class_=StaticArgs, allow_None=True, constant=True, doc='''
             The specifier in which a lookup is performed to find the unique
             matching specification. The index must be longer than the operand
             and the values of the index_key must be unique.''')

    index_key  = param.String(default=None, allow_None=True, constant=True, doc='''
             The common key in both the index and the operand used to
             index the former specifications into the latter.''')

    def __init__(self, operand, index, index_key, fp_precision=None, **kwargs):

        if False in [isinstance(operand, StaticArgs), isinstance(index, StaticArgs)]:
            raise Exception('Can only index two Static Argument specifiers')

        max_precision = max(operand.fp_precision, index.fp_precision)
        fp_precision =  max_precision if fp_precision is None else fp_precision

        specs = self._index(operand.specs, index.specs, index_key)
        super(Indexed,self).__init__(specs, fp_precision=fp_precision, index_key=index_key,
                                     index=index, operand=operand, **kwargs)
        self.pprint_args(['operand', 'index', 'index_key'],[])

    def _index(self, specs, index_specs, index_key):
        keys = [spec[index_key] for spec in specs]
        index_keys = [spec[index_key] for spec in index_specs]

        if len(index_keys) != len(set(index_keys)):
            raise Exception("Keys in index must all be unique")
        if not set(keys).issubset(set(index_keys)):
            raise Exception("Keys in specifier must be subset of keys in index.")

        spec_items = zip(keys, specs)
        index_idxmap = dict(zip(index_keys, range(len(index_keys))))

        return [dict(v, **index_specs[index_idxmap[k]]) for (k,v) in spec_items]

class FilePattern(StaticArgs):
    """
    A FilePattern specifier allows files to be located via an extended form of
    globbing. For example, you can find the absolute filenames of all npz files
    in the data subdirectory (relative to the root) that start with the filename
    'timeseries' use the pattern 'data/timeseries*.npz'.

    In addition to globbing supported by the glob module, patterns can extract
    metadata from filenames using a subset of the Python format specification
    syntax. To illustrate, you can use 'data/timeseries-{date}.npz' to record
    the date strings associated with matched files. Note that a particular named
    fields can only be used in a particular pattern once.

    By default metadata is extracted as strings but format types are supported
    in the usual manner eg. 'data/timeseries-{day:d}-{month:d}.npz' will extract
    the day and month from the filename as integers. Only field names and types
    are recognised with all other format specification ignored. Type codes
    supported: 'd', 'b', 'o', 'x', 'e','E','f', 'F','g', 'G', 'n' (otherwise
    result is a string).

    Note that ordering is determined via ascending alphanumeric sort and that
    actual filenames should not include any globbing characters, namely: '?','*','['
    and ']' (general good practice for filenames).
    """

    key = param.String(default=None, allow_None=True, constant=True, doc='''
             The key name given to the matched file path strings.''')

    pattern = param.String(default=None, allow_None=True, constant=True,
              doc='''The pattern files are to be searched against.''')

    root = param.String(default=None, allow_None=True, constant=True, doc='''
             The root directory from which patterns are to be loaded.  If set to
             None, normalize_path.prefix is used (os.getcwd() by default).''')

    @classmethod
    def directory(cls, directory, root=None, extension=None, **kwargs):
        """
        Load all the files in a given directory. Only files with the given file
        extension are loaded if the extension is specified. The given kwargs are
        passed through to the normal constructor.
        """
        root = param.normalize_path.prefix if root is None else root
        suffix = '' if extension is None else '.' + extension.rsplit('.')[-1]
        pattern = directory + os.sep + '*' + suffix
        key = os.path.join(root, directory,'*').rsplit(os.sep)[-2]
        format_parse = list(string.Formatter().parse(key))
        if not all([el is None for el in zip(*format_parse)[1]]):
            raise Exception('Directory cannot contain format field specifications')
        return cls(key, pattern, root, **kwargs)

    def __init__(self, key, pattern, root=None, **kwargs):
        root = param.normalize_path.prefix if root is None else root
        specs = self._load_expansion(key, root, pattern)
        updated_specs = self._load_file_metadata(specs, key, **kwargs)
        super(FilePattern, self).__init__(updated_specs, key=key, pattern=pattern,
                                          root=root, **kwargs)
        if len(updated_specs) == 0:
            print("%r: No matches found." % self)
        self.pprint_args(['key', 'pattern'], ['root'])

    def fields(self):
        """
        Return the fields specified in the pattern using Python's formatting
        mini-language.
        """
        parse = list(string.Formatter().parse(self.pattern))
        return [f for f in zip(*parse)[1] if f is not None]

    def _load_file_metadata(self, specs, key, **kwargs):
        """
        Hook to allow a subclass to load metadata from the located files.
        """
        return specs

    def _load_expansion(self, key, root, pattern):#, lexsort):
        """
        Loads the files that match the given pattern.
        """
        path_pattern = os.path.join(root, pattern)
        expanded_paths = self._expand_pattern(path_pattern)

        specs=[]
        for (path, tags) in expanded_paths:
            rootdir = path if os.path.isdir(path) else os.path.split(path)[0]
            filelist = [os.path.join(path,f) for f in os.listdir(path)] if os.path.isdir(path) else [path]
            for filepath in filelist:
                specs.append(dict(tags,**{key:filepath}))

        return sorted(specs, key=lambda s: s[key])

    def _expand_pattern(self, pattern):
        """
        From the pattern decomposition, finds the absolute paths matching the pattern.
        """
        (globpattern, regexp, fields, types) = self._decompose_pattern(pattern)
        filelist = glob.glob(globpattern)
        expansion = []

        for fname in filelist:
            if fields == []:
                expansion.append((fname, {}))
                continue
            match = re.match(regexp, fname)
            if match is None: continue
            match_items = match.groupdict().items()
            tags = dict((k,types.get(k, str)(v)) for (k,v) in match_items)
            expansion.append((fname, tags))

        return expansion

    def _decompose_pattern(self, pattern):
        """
        Given a path pattern with format declaration, generates a four-tuple
        (glob_pattern, regexp pattern, fields, type map)
        """
        sep = '~lancet~sep~'
        float_codes = ['e','E','f', 'F','g', 'G', 'n']
        typecodes = dict([(k,float) for k in float_codes] + [('b',bin), ('d',int), ('o',oct), ('x',hex)])
        parse = list(string.Formatter().parse(pattern))
        text, fields, codes, _ = zip(*parse)

        # Finding the field types from format string
        types = []
        for (field, code) in zip(fields, codes):
            if code in ['', None]: continue
            constructor =  typecodes.get(code[-1], None)
            if constructor: types += [(field, constructor)]

        stars =  ['' if not f else '*' for f in fields]
        globpattern = ''.join(text+star for (text,star) in zip(text, stars))

        refields = ['' if not f else sep+('(?P<%s>.*?)'% f)+sep for f in fields]
        parts = ''.join(text+group for (text,group) in zip(text, refields)).split(sep)
        for i in range(0, len(parts), 2): parts[i] = re.escape(parts[i])

        return globpattern, ''.join(parts).replace('\\*','.*'), list(f for f in fields if f), dict(types)

class LexSorted(StaticArgs):
    """
    Argument specifiers normally have a clearly defined but implicit, default
    orderings. Sometimes a different ordering is desired, typically for
    inspecting the structure of the specifier in some way
    (ie. viewing). Applying LexSorted to a specifier allows the desired
    ordering to be achieved.

    The lexical sort order is specified in the 'order' parameter which takes a
    list of strings. Each string is a key name prefixed by '+' or '-' for
    ascending and descending sort respectively. If the key is not found in the
    operand's set of varying keys, it is ignored.

    To illustrate, if order=['+id', '-time'] then the specifier would be sorted
    by ascending by 'id'value but where id values are equal, it would be sorted
    by descending 'time' value.
    """
    operand = param.ClassSelector(default=None, class_=StaticArgs, allow_None=True, constant=True, doc='''
              The source specifier which is to be lexically sorted.''')

    order = param.List(default=[], constant=True, doc='''
             An ordered list of annotated keys for lexical sorting. An annotated
             key is the usual key name prefixed with either '+' (for ascending
             sort) or '-' (for descending sort). By default, no sorting is applied.''')

    def __init__(self, operand, order=[], **kwargs):
        specs = self._lexsort(operand, order)
        super(LexSorted, self).__init__(specs, operand=operand, order=order, **kwargs)
        self.pprint_args(['operand','order'],[])

    def _lexsort(self, operand, order=[]):
        """
        A lexsort is specified using normal key string prefixed by '+' (for
        ascending) or '-' for (for descending).

        Note that in Python 2, if a key is missing, None is returned (smallest
        Python value). In Python 3, an Exception will be raised regarding
        comparison of heterogenous types.
        """

        specs = operand.specs[:]
        if not all(el[0] in ['+', '-'] for el in order):
            raise Exception("Please prefix sort keys with either '+' (for ascending) or '-' for descending")

        sort_cycles = [(el[1:], True if el[0]=='+' else False) for el in reversed(order)
                       if el[1:] in operand.varying_keys()]


        for (key, ascending) in sort_cycles:
            specs = sorted(specs, key=lambda s: s.get(key, None), reverse=(not ascending))
        return specs


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
        self.pprint_args(['first', 'second'],[], infix_operator='+')

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

        self.pprint_args(['first', 'second'],[], infix_operator='*')

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
        constant_keys and constant_items.
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
                'constant_keys':      arg_specifier.constant_keys(),
                'constant_items':     arg_specifier.constant_items()}

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
                cmd_lines = ['%d: %s\n' % (i, ' '.join(qcmds)) for (i,qcmds) in enumerate(quoted_cmds)]
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

    metadata = param.Dict(default={}, doc='''
              Metadata information to add to the info file.''')

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
        self._spec_log += specs # This should be removed
        log_path = os.path.join(self.root_directory, ("%s.log" % self.batch_name))
        Log.write_log(log_path, [spec for (_, spec) in specs], allow_append=True)

    def record_info(self, setup_info=None):
        """
        All launchers should call this method to write the info file at the end
        of the launch. The info file saves the given setup_info, usually the
        launch dict returned by _setup_launch. The file is written to the
        root_directory. When called without setup_info, the existing info file
        is being updated with the end-time.
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

        return {'root_directory':    self.root_directory,
                'timestamp':         self.timestamp,
                'varying_keys':      self.arg_specifier.varying_keys(),
                'constant_keys':     self.arg_specifier.constant_keys(),
                'constant_items':     self.arg_specifier.constant_items(),
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

        self.record_info(launchinfo)

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

        self.record_info(self._launchinfo)

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

class applying(param.Parameterized):
    """
    Decorator to invoke Python code (callables) with a specifier, optionally
    creating a log of the arguments used.  By default data is passed in as
    keywords but positional arguments can be specified using the 'args'
    parameter.

    Automatically accumulates the return values of any callable (functions or
    classes). The return value is an instance of this class which may be called
    without arguments to repeat the last operation or bound to another function
    with the same call signature to call that instead.

    values = applying(ListArgs('value',[1,2,3]))

    values(lambda value: value +1)
    values(lambda value: value**2)
    values() # Repeats the last function set

    values.accumulator
    ... [2, 3, 4, 1, 4, 9, 1, 4, 9]

    May also be used as a decorator to wrap a single function:

    @applying(ListArgs('value',[1,2,3,4]))
    def add_one(value=None):
        return value +1

    add_one.accumulator
    ... [2, 3, 4]

    Dynamic specifiers may be updated as necessary with the update_fn parameter.
    """

    specifier = param.ClassSelector(default=None, allow_None=True, constant=True, class_=StaticArgs,
               doc='''The specifier from which the positional and keyword
                arguments are to be derived.''')

    args = param.List(default=[], constant=True, doc='''The list of positional arguments to generate.''')

    callee = param.Callable(doc='''The function that is to be applied.''')

    log_path = param.String(default=None, allow_None=True, doc='''
              Optional path to a log file for recording the list of arguments used.''')

    update_fn = param.Callable(default=lambda spec, values: None, doc='''
                 Hook to call to update dynamic specifiers as necessary.  This
                 callable takes two arguments, first the specifier that needs
                 updating and the list of accumulated values from the current
                 group of results.''')

    accumulator = param.List(default=[], doc='''Accumulates the return values of the callable.''')

    def __init__(self, specifier, **kwargs):
        super(applying, self).__init__(specifier=specifier, **kwargs)

    @property
    def kwargs(self):
        all_keys = self.specifier.constant_keys() + self.specifier.varying_keys()
        return [k for k in all_keys if k not in self.args]

    def _args_kwargs(self, specs, args):
        """
        Separates out args from kwargs given a list of non-kwarg arguments.
        When the args list is empty, kwargs alone are returned.
        """
        if args ==[]: return specs
        arg_list = [v for (k,v) in specs.items() if k in args]
        kwarg_dict = dict((k,v) for (k,v) in specs.items() if (k not in args))
        return (arg_list, kwarg_dict)

    def __call__(self, fn=None):
        if fn is not None:
            self.callee = fn
            return self

        if self.callee is None:
            print('No callable specified.')
            return self

        if self.log_path and os.path.isfile(self.log_path):
            raise Exception('Log %r already exists.' % self.log_path)

        log = []
        for concurrent_group in self.specifier:
            concurrent_values = []
            for specs in concurrent_group:
                value = self.callee(**specs)
                concurrent_values.append(value)
                log.append(specs)

            self.update_fn(self.specifier, concurrent_values)
            self.accumulator.extend(concurrent_values)

        if self.log_path:
            Log.write_log(self.log_path, log, allow_append=False)
        return self

    def __repr__(self):
        arg_list = ['%r' % self.specifier,
                'args=%s' % self.args if self.args else None,
                'accumulator=%s' % self.accumulator]
        arg_str = ','.join(el for el in arg_list if el is not None)
        return 'applying(%s)' % arg_str

    def __str__(self):
        arg_list = ['args=%r' % self.args if self.args else None,
                    'accumulator=%r' % self.accumulator]
        arg_str = ',\n   ' + ',\n    '.join(el for el in arg_list if el is not None)
        return 'applying(\n   specifier=%s%s\n)' % (self.specifier._pprint(level=2), arg_str)

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
    parameter is the proper way of executing code after the Launcher exits.
    """

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

    launch_fn = param.Callable(doc='''The function that is to be applied.''')

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

    def __call__(self, fn=None):

        if fn is not None:
            self.launch_fn = fn
            return self

        if self.main_script and self.launch_fn.__module__ != '__main__': return False

        # Resuming launch as necessary
        if self.launcher_class.resume_launch(): return False

        # Setting the output directory via param
        if self.output_directory is not None:
            param.normalize_path.prefix = self.output_directory

        # Calling the wrapped function with appropriate arguments
        kwargs_list = [{}] if (self.launch_args is None) else self.launch_args.specs
        lvals = [self.launch_fn(**kwargs_list[0])]
        if self.launch_args is not None:
            self.launcher_class.timestamp = self._get_launcher(lvals[0]).timestamp
            lvals += [self.launch_fn(**kwargs) for kwargs in kwargs_list[1:]]

        # Cross checks
        for (accessor, checker) in self._cross_checks:
            checker([accessor(lval) for lval in lvals])

        if self.review:
            # Run review of launch args only if necessary
            if self.launch_args is not None:
                proceed = self.review_args(self.launch_args, heading='Lancet Meta Parameters')
                if not proceed: return False

            for (count, lval) in enumerate(lvals):
                proceed = all(r(access(lval)) for (access, r) in self._reviewers)
                if not proceed:
                    print("Aborting launch.")
                    return False

                if len(lvals)!= 1 and count < len(lvals)-1:
                    skip_remaining = self.input_options(['Y', 'n','quit'],
                                     '\nSkip remaining reviews?', default='y')
                    if skip_remaining == 'quit': return False
                    if skip_remaining == 'y': break

            self._process_launchers(lvals)

            if self.input_options(['y','N'], 'Execute?', default='n') != 'y':
                return False

        # Run configure hook. Exit if any configuration fails
        if not all([self.configure_launch(lval) for lval in lvals]): return False

        for lval in lvals:
            launcher =  self._get_launcher(lval)
            print("== Launching  %s ==" % launcher.batch_name)
            launcher.launch()

        return True

    def _process_launchers(self, lvals):
        pass

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
        items = '\n'.join(['%s = %r' % (k,v) for (k,v) in arg_specifier.constant_items()])
        print("Constant Items:\n\n%s\n" % items)
        print("Definition:\n%s" % arg_specifier)

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

    def __repr__(self):
        arg_list = ['%s' % self.launcher_class.__name__,
                    '%r' % self.output_directory,
                    'launch_args=%r' % self.launch_args if self.launch_args else None,
                    'review=False' if not self.review else None,
                    'main_script=False' if not self.main_script else None ]
        arg_str = ','.join(el for el in arg_list if el is not None)
        return 'review_and_launch(%s)' % arg_str

    def __str__(self):
        arg_list = ['launcher_class=%s' % self.launcher_class.__name__,
                    'output_directory=%r' % self.output_directory,
                    'launch_args=%s' % self.launch_args._pprint(level=2) if self.launch_args else None,
                    'review=False' if not self.review else None,
                    'main_script=False' if not self.main_script else None ]
        arg_str = ',\n   '.join(el for el in arg_list if el is not None)
        return 'review_and_launch(\n   %s\n)' % arg_str

