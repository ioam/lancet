#
# Lancet core
#

import os, time, itertools, copy
import re, glob, string, re
import json

import param

try:
    import numpy as np
    np_ftypes = np.sctypes['float']
except:
    np, np_ftypes = None, []

from collections import defaultdict

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
        self._pprint_args = ([],[],None,{})
        super(BaseArgs,self).__init__(**params)
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

        if self.dynamic or other.dynamic: return DynamicConcatenate(self,other)
        else:                             return StaticConcatenate(self,other)

    def __mul__(self, other):
        """
        Takes the cartesian product of two argument specifiers. See
        StaticCartesianProduct and DynamicCartesianProduct documentation.
        """
        if not other: return []
        assert not (self.dynamic and other.dynamic), \
            'Cannot take Cartesian product two dynamic specifiers.'

        if self.dynamic or other.dynamic: return DynamicCartesianProduct(self, other)
        else:                             return StaticCartesianProduct(self, other)

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

