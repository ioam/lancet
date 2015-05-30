#
# Lancet core
#

import os, itertools, copy
import re, glob, string
import json

import param

try:
    import numpy as np
    np_ftypes = np.sctypes['float']
except:
    np, np_ftypes = None, []

try:    from pandas import DataFrame
except: DataFrame = None # pyflakes:ignore (try/except import)

try: from holoviews import Table
except: Table = None     # pyflakes:ignore (try/except import)

from collections import defaultdict, OrderedDict

float_types = [float] + np_ftypes
def identityfn(x): return x
def fp_repr(x):    return str(x) if (type(x) in float_types) else repr(x)

def set_fp_precision(value):
    """
    Function to set the floating precision across lancet.
    """
    Arguments.set_default('fp_precision', value)

#=====================#
# Argument Specifiers #
#=====================#

class PrettyPrinted(object):
    """
    A mixin class for generating pretty-printed representations.
    """

    def pprint_args(self, pos_args, keyword_args, infix_operator=None, extra_params={}):
        """
        Method to define the positional arguments and keyword order
        for pretty printing.
        """
        if infix_operator and not (len(pos_args)==2 and keyword_args==[]):
            raise Exception('Infix format requires exactly two'
                            ' positional arguments and no keywords')
        (kwargs,_,_,_) = self._pprint_args
        self._pprint_args = (keyword_args + kwargs, pos_args, infix_operator, extra_params)

    def _pprint(self, cycle=False, flat=False, annotate=False, onlychanged=True, level=1, tab = '   '):
        """
        Pretty printer that prints only the modified keywords and
        generates flat representations (for repr) and optionally
        annotates the top of the repr with a comment.
        """
        (kwargs, pos_args, infix_operator, extra_params) = self._pprint_args
        (br, indent)  = ('' if flat else '\n', '' if flat else tab * level)
        prettify = lambda x: isinstance(x, PrettyPrinted) and not flat
        pretty = lambda x: x._pprint(flat=flat, level=level+1) if prettify(x) else repr(x)

        params = dict(self.get_param_values())
        show_lexsort = getattr(self, '_lexorder', None) is not None
        modified = [k for (k,v) in self.get_param_values(onlychanged=onlychanged)]
        pkwargs = [(k, params[k])  for k in kwargs if (k in modified)] + list(extra_params.items())
        arg_list = [(k,params[k]) for k in pos_args] + pkwargs

        lines = []
        if annotate: # Optional annotating comment
            len_ckeys, len_vkeys = len(self.constant_keys), len(self.varying_keys)
            info_triple = (len(self),
                           ', %d constant key(s)' % len_ckeys if len_ckeys else '',
                           ', %d varying key(s)'  % len_vkeys if len_vkeys else '')
            annotation = '# == %d items%s%s ==\n' % info_triple
            lines = [annotation]

        if show_lexsort: lines.append('(')
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

        if show_lexsort:
            lines.append(').lexsort(%s)' % ', '.join(repr(el) for el in self._lexorder))
        return ''.join(lines)

    def __repr__(self):
        return self._pprint(flat=True, onlychanged=False)

    def __str__(self):
        return self._pprint()



class Arguments(PrettyPrinted, param.Parameterized):
    """
    The abstract, base class that defines the core interface and
    methods for all members of the Arguments family of classes,
    including either the simple, static members of Args below, or the
    sophisticated parameter exploration algorithms subclassing from
    DynamicArgs defined in dynamic.py.

    The Args subclass may be used directly and forms the root of one
    family of classes that have statically defined or precomputed
    argument sets (defined below). The second subfamily are the
    DynamicArgs, designed to allow more sophisticated, online
    parameter space exploration techniques such as hill climbing,
    bisection search, genetic algorithms and so on.
    """

    fp_precision = param.Integer(default=4, constant=True, doc='''
         The floating point precision to use for floating point
         values.  Unlike other basic Python types, floats need care
         with their representation as you only want to display up to
         the precision actually specified. A floating point precision
         of 0 casts number to integers before representing them.''')

    def __init__(self, **params):
        self._pprint_args = ([],[],None,{})
        self.pprint_args([],['fp_precision', 'dynamic'])
        super(Arguments,self).__init__(**params)
        # Some types cannot be sorted easily (e.g. numpy arrays)
        self.unsortable_keys = []

    def __iter__(self): return self

    def __contains__(self, value):
        return value in (self.constant_keys + self.varying_keys)

    @classmethod
    def spec_formatter(cls, spec):
        " Formats the elements of an argument set appropriately"
        return type(spec)((k, str(v)) for (k,v) in spec.items())

    @property
    def constant_keys(self):
        """
        Returns the list of parameter names whose values are constant
        as the argument specifier is iterated.  Note that the union of
        constant and varying_keys should partition the entire set of
        keys in the case where there are no unsortable keys.
        """
        raise NotImplementedError

    @property
    def constant_items(self):
        """
        Returns the set of constant items as a list of tuples. This
        allows easy conversion to dictionary format. Note, the items
        should be supplied in the same key ordering as for
        constant_keys for consistency.
        """
        raise NotImplementedError

    @property
    def varying_keys(self):
        """
        Returns the list of parameters whose values vary as the
        argument specifier is iterated.  Whenever it is possible, keys
        should be sorted from those slowest to faster varying and
        sorted alphanumerically within groups that vary at the same
        rate.
        """
        raise NotImplementedError

    def round_floats(self, specs, fp_precision):
        _round_float = lambda v, fp: np.round(v, fp) if (type(v) in np_ftypes) else round(v, fp)
        _round = (lambda v, fp: int(v)) if fp_precision==0  else _round_float
        return (dict((k, _round(v, fp_precision) if (type(v) in float_types) else v)
                     for (k,v) in spec.items()) for spec in specs)

    def __next__(self):
        """
        Called to get a list of specifications: dictionaries with
        parameter name keys and string values.
        """
        raise StopIteration

    next = __next__

    def copy(self):
        """
        Convenience method to avoid using the specifier without
        exhausting it.
        """
        return copy.copy(self)

    def _collect_by_key(self,specs):
        """
        Returns a dictionary like object with the lists of values
        collapsed by their respective key. Useful to find varying vs
        constant keys and to find how fast keys vary.
        """
        # Collect (key, value) tuples as list of lists, flatten with chain
        allkeys = itertools.chain.from_iterable(
            [[(k, run[k]) for k in run] for run in specs])
        collection = defaultdict(list)
        for (k,v) in allkeys: collection[k].append(v)
        return collection

    def _operator(self, operator, other):
        identities = [isinstance(el, Identity) for el in [self, other]]
        if  not any(identities): return operator(self,other)
        if all(identities):      return Identity()
        elif identities[1]:      return self
        else:                    return other

    def __add__(self, other):
        """
        Concatenates two argument specifiers.
        """
        return self._operator(Concatenate, other)

    def __mul__(self, other):
        """
        Takes the Cartesian product of two argument specifiers.
        """
        return self._operator(CartesianProduct, other)

    def _cartesian_product(self, first_specs, second_specs):
        """
        Takes the Cartesian product of the specifications. Result will
        contain N specifications where N = len(first_specs) *
        len(second_specs) and keys are merged.
        Example: [{'a':1},{'b':2}] * [{'c':3},{'d':4}] =
        [{'a':1,'c':3},{'a':1,'d':4},{'b':2,'c':3},{'b':2,'d':4}]
        """
        return  [ dict(zip(
                          list(s1.keys()) + list(s2.keys()),
                          list(s1.values()) + list(s2.values())
                      ))
                 for s1 in first_specs for s2 in second_specs ]

    def summary(self):
        """
        A succinct summary of the argument specifier. Unlike the repr,
        a summary does not have to be complete but must supply the
        most relevant information about the object to the user.
        """
        print("Items: %s" % len(self))
        varying_keys = ', '.join('%r' % k for k in self.varying_keys)
        print("Varying Keys: %s" % varying_keys)
        items = ', '.join(['%s=%r' % (k,v)
                           for (k,v) in self.constant_items])
        if self.constant_items:
            print("Constant Items: %s" % items)


class Identity(Arguments):
    """
    The identity element for any Arguments object 'args' under the *
    operator (CartesianProduct) and + operator (Concatenate). The
    following identities hold:

    args is (Identity() * args)
    args is (args * Identity())

    args is (Identity() + args)
    args is (args + Identity())

    Note that the empty Args() object can also fulfill the role of
    Identity under the addition operator.
    """

    fp_precision = param.Integer(default=None, allow_None=True,
                                 precedence=(-1), constant=True, doc='''
       fp_precision is disabled as Identity() never contains any
       arguments.''')

    def __eq__(self, other): return isinstance(other, Identity)
    def __repr__(self): return "Identity()"
    def __str__(self): return repr(self)
    def __nonzero__(self): raise ValueError("The boolean value of Identity is undefined")
    def __bool__(self): raise ValueError("The boolean value of Identity is undefined")


class Args(Arguments):
    """
    An Arguments class that supports statically specified or
    precomputed argument sets. It may be used directly to specify
    argument values but also forms the base class for a family of more
    specific static Argument classes. Each subclass is less flexible
    and general but allows arguments to be easily and succinctly
    specified. For instance, the Range subclass allows parameter
    ranges to be easily declared.

    The constructor of Args accepts argument definitions in two
    different formats. The keyword format allows constant arguments to
    be specified directly and easily. For instance:

    >>> v1 = Args(a=2, b=3)
    >>> v1
    Args(fp_precision=4,a=2,b=3)

    The alternative input format takes an explicit list of the
    argument specifications:

    >>> v2 = Args([{'a':2, 'b':3}]) # Equivalent behaviour to above
    >>> v1.specs  == v2.specs
    True

    This latter format is completely flexible and general, allowing
    any arbitrary list of arguments to be specified as desired. This
    is not generally recommended however as the structure of a
    parameter space is often expressed more clearly by composing
    together simpler, more succinct Args objects with the
    CartesianProduct (*) or Concatenation (+) operators.
    """

    specs = param.List(default=[], constant=True, doc='''
          The static list of specifications (ie. dictionaries) to be
          returned by the specifier. Float values are rounded
          according to fp_precision.''')

    def __init__(self, specs=None, fp_precision=None, **params):
        if fp_precision is None: fp_precision = Arguments.fp_precision
        raw_specs, params, explicit = self._build_specs(specs, params, fp_precision)
        super(Args, self).__init__(fp_precision=fp_precision, specs=raw_specs, **params)

        self._lexorder = None
        if explicit:
            self.pprint_args(['specs'],[])
        else: # Present in kwarg format
            self.pprint_args([], self.constant_keys, None,
                             OrderedDict(sorted(self.constant_items)))

    def _build_specs(self, specs, kwargs, fp_precision):
        """
        Returns the specs, the remaining kwargs and whether or not the
        constructor was called with kwarg or explicit specs.
        """
        if specs is None:
            overrides = param.ParamOverrides(self, kwargs,
                                             allow_extra_keywords=True)
            extra_kwargs = overrides.extra_keywords()
            kwargs = dict([(k,v) for (k,v) in kwargs.items()
                           if k not in extra_kwargs])
            rounded_specs = list(self.round_floats([extra_kwargs],
                                                   fp_precision))

            if extra_kwargs=={}: return [], kwargs, True
            else:                return rounded_specs, kwargs, False

        return list(self.round_floats(specs, fp_precision)), kwargs, True

    def __iter__(self):
        self._exhausted = False
        return self

    def __next__(self):
        if self._exhausted:
            raise StopIteration
        else:
            self._exhausted=True
            return self.specs

    next = __next__

    def _unique(self, sequence, idfun=repr):
        """
        Note: repr() must be implemented properly on all objects. This
        is implicitly assumed by Lancet when Python objects need to be
        formatted to string representation.
        """
        seen = {}
        return [seen.setdefault(idfun(e),e) for e in sequence
                if idfun(e) not in seen]

    def show(self, exclude=[]):
        """
        Convenience method to inspect the available argument values in
        human-readable format. The ordering of keys is determined by
        how quickly they vary.

        The exclude list allows specific keys to be excluded for
        readability (e.g. to hide long, absolute filenames).
        """
        ordering = self.constant_keys + self.varying_keys
        spec_lines = [', '.join(['%s=%s' % (k, s[k]) for k in ordering
                                 if (k in s) and (k not in exclude)])
                      for s in self.specs]
        print('\n'.join(['%d: %s' % (i,l) for (i,l) in enumerate(spec_lines)]))


    def lexsort(self, *order):
        """
        The lexical sort order is specified by a list of string
        arguments. Each string is a key name prefixed by '+' or '-'
        for ascending and descending sort respectively. If the key is
        not found in the operand's set of varying keys, it is ignored.
        """
        if order == []:
            raise Exception("Please specify the keys for sorting, use"
                            "'+' prefix for ascending,"
                            "'-' for descending.)")

        if not set(el[1:] for el in order).issubset(set(self.varying_keys)):
            raise Exception("Key(s) specified not in the set of varying keys.")

        sorted_args = copy.deepcopy(self)
        specs_param = sorted_args.params('specs')
        specs_param.constant = False
        sorted_args.specs = self._lexsorted_specs(order)
        specs_param.constant = True
        sorted_args._lexorder = order
        return sorted_args

    def _lexsorted_specs(self, order):
        """
        A lexsort is specified using normal key string prefixed by '+'
        (for ascending) or '-' for (for descending).

        Note that in Python 2, if a key is missing, None is returned
        (smallest Python value). In Python 3, an Exception will be
        raised regarding comparison of heterogenous types.
        """
        specs = self.specs[:]
        if not all(el[0] in ['+', '-'] for el in order):
            raise Exception("Please specify the keys for sorting, use"
                            "'+' prefix for ascending,"
                            "'-' for descending.)")

        sort_cycles = [(el[1:], True if el[0]=='+' else False)
                       for el in reversed(order)
                       if el[1:] in self.varying_keys]

        for (key, ascending) in sort_cycles:
            specs = sorted(specs, key=lambda s: s.get(key, None),
                           reverse=(not ascending))
        return specs

    @property
    def constant_keys(self):
        collection = self._collect_by_key(self.specs)
        return [k for k in sorted(collection) if
                (len(self._unique(collection[k])) == 1)]

    @property
    def constant_items(self):
        collection = self._collect_by_key(self.specs)
        return [(k,collection[k][0]) for k in self.constant_keys]

    @property
    def varying_keys(self):
        collection = self._collect_by_key(self.specs)
        constant_set = set(self.constant_keys)
        unordered_varying = set(collection.keys()).difference(constant_set)
        # Finding out how fast keys are varying
        grouplens = [(len([len(list(y)) for (_,y)
                           in itertools.groupby(collection[k])]),k)
                     for k in collection
                     if (k not in self.unsortable_keys)]
        varying_counts = [(n,k) for (n,k) in sorted(grouplens) if (k in unordered_varying)]
        # Grouping keys with common frequency alphanumerically (desired behaviour).
        ddict = defaultdict(list)
        for (n,k) in varying_counts: ddict[n].append(k)
        alphagroups = [sorted(ddict[k]) for k in sorted(ddict)]
        return [el for group in alphagroups for el in group] + sorted(self.unsortable_keys)

    @property
    def dframe(self):
        return DataFrame(self.specs) if DataFrame else "Pandas not available"

    @property
    def table(self):
        if not Table:
            return "HoloViews Table not available"
        keys =  self.varying_keys + self.constant_keys
        items = [(tuple([spec[k] for k  in keys]),()) for spec in self.specs]
        return Table(items, key_dimensions=keys, value_dimensions=[])


    def __len__(self): return len(self.specs)



class Concatenate(Args):
    """
    Concatenate is the sequential composition of two specifiers. The
    specifier created by the compositon (firsts + second) generates
    the arguments in first followed by the arguments in second.
    """

    first = param.ClassSelector(default=None, class_=Args, allow_None=True,
       constant=True, doc='''The first specifier in the concatenation.''')

    second = param.ClassSelector(default=None, class_=Args, allow_None=True,
       constant=True, doc='''The second specifier in the concatenation.''')

    def __init__(self, first, second):

        max_precision = max(first.fp_precision, second.fp_precision)
        specs = first.specs + second.specs
        super(Concatenate, self).__init__(specs, fp_precision=max_precision,
                                                first=first, second=second)
        self.pprint_args(['first', 'second'],[], infix_operator='+')


class CartesianProduct(Args):
    """
    CartesianProduct is the Cartesian product of two specifiers. The
    specifier created by the compositon (firsts * second) generates
    the cartesian produce of the arguments in first followed by the
    arguments in second. Note that len(first * second) =
    len(first)*len(second)
    """

    first = param.ClassSelector(default=None, class_=Args, allow_None=True,
       constant=True, doc='''The first specifier in the Cartesian product.''')

    second = param.ClassSelector(default=None, class_=Args, allow_None=True,
       constant=True, doc='''The second specifier in the Cartesian product.''')

    def __init__(self, first, second):

        max_precision = max(first.fp_precision, second.fp_precision)
        specs = self._cartesian_product(first.specs, second.specs)

        overlap = (set(first.varying_keys + first.constant_keys)
                   &  set(second.varying_keys + second.constant_keys))
        assert overlap == set(), ('Sets of keys cannot overlap'
                                  'between argument specifiers'
                                  'in cartesian product.')
        super(CartesianProduct, self).__init__(specs, fp_precision=max_precision,
                                               first=first, second=second)
        self.pprint_args(['first', 'second'],[], infix_operator='*')


class Range(Args):
    """
    Range generates an argument from a numerically interpolated range
    which is linear by default. An optional function can be specified
    to sample a numeric range with regular intervals.
    """

    key = param.String(default='', constant=True, doc='''
         The key assigned to the values computed over the numeric range.''')

    start_value =  param.Number(default=None, allow_None=True, constant=True,
        doc='''The starting numeric value of the range.''')

    end_value = param.Number(default=None, allow_None=True, constant=True,
       doc='''The ending numeric value of the range (inclusive).''')

    steps = param.Integer(default=2, constant=True, bounds=(1,None),
       doc='''The number of steps to interpolate over. Default is 2
         which returns the start and end values without interpolation.''')

    # Can't this be a lambda?
    mapfn = param.Callable(default=identityfn, constant=True, doc='''
         The function to be mapped across the linear range. The
         identity function is used by by default''')

    def __init__(self, key, start_value, end_value, steps=2, mapfn=identityfn, **params):

        values = self.linspace(start_value, end_value, steps)
        specs = [{key:mapfn(val)} for val in values ]

        super(Range, self).__init__(specs, key=key, start_value=start_value,
                                         end_value=end_value, steps=steps,
                                         mapfn=mapfn, **params)
        self.pprint_args(['key', 'start_value'], ['end_value', 'steps'])

    def linspace(self, start, stop, n):
        """ Simple replacement for numpy linspace"""
        if n == 1: return [start]
        L = [0.0] * n
        nm1 = n - 1
        nm1inv = 1.0 / nm1
        for i in range(n):
            L[i] = nm1inv * (start*(nm1 - i) + stop*i)
        return L

class List(Args):
    """
    An argument specifier that takes its values from a given list.
    """

    values = param.List(default=[], constant=True, doc='''
         The list values that are to be returned by the specifier''')

    key = param.String(default='default', constant=True, doc='''
         The key assigned to the elements of the supplied list.''')

    def __init__(self, key, values, **params):
        specs = [{key:val} for val in values]
        super(List, self).__init__(specs, key=key, values=values, **params)
        self.pprint_args(['key', 'values'], [])


class Log(Args):
    """
    Specifier that loads arguments from a log file in task id (tid)
    order.  This wrapper class allows a concise representation of file
    logs with the option of adding the task id to the loaded
    specifications.

    For full control over the arguments, you can use this class to
    create a fully specified Args object as follows:

    Args(Log.extract_log(<log_file>).values()),
    """

    log_path = param.String(default=None, allow_None=True, constant=True,
         doc='''The relative or absolute path to the log file. If a
              relative path is given, the absolute path is computed
              relative to os.getcwd().''')

    tid_key = param.String(default='tid', constant=True, allow_None=True,
         doc='''If not None, the key given to the tid values included
               in the loaded specifications. If None, the tid number
               is ignored.''')

    @staticmethod
    def extract_log(log_path, dict_type=dict):
        """
        Parses the log file generated by a launcher and returns
        dictionary with tid keys and specification values.

        Ordering can be maintained by setting dict_type to the
        appropriate constructor (i.e. OrderedDict). Keys are converted
        from unicode to strings for kwarg use.
        """
        log_path = (log_path if os.path.isfile(log_path)
                    else os.path.join(os.getcwd(), log_path))
        with open(log_path,'r') as log:
            splits = (line.split() for line in log)
            uzipped = ((int(split[0]), json.loads(" ".join(split[1:]))) for split in splits)
            szipped = [(i, dict((str(k),v) for (k,v) in d.items())) for (i,d) in uzipped]
        return dict_type(szipped)

    @staticmethod
    def write_log(log_path, data, allow_append=True):
        """
        Writes the supplied specifications to the log path. The data
        may be supplied as either as a an Args or as a list of
        dictionaries.

        By default, specifications will be appropriately appended to
        an existing log file. This can be disabled by setting
        allow_append to False.
        """
        append = os.path.isfile(log_path)
        islist = isinstance(data, list)

        if append and not allow_append:
            raise Exception('Appending has been disabled'
                            ' and file %s exists' % log_path)

        if not (islist or isinstance(data, Args)):
            raise Exception('Can only write Args objects or dictionary'
                            ' lists to log file.')

        specs = data if islist else data.specs
        if not all(isinstance(el,dict) for el in specs):
            raise Exception('List elements must be dictionaries.')

        log_file = open(log_path, 'r+') if append else open(log_path, 'w')
        start = int(log_file.readlines()[-1].split()[0])+1 if append else 0
        ascending_indices = range(start, start+len(data))

        log_str = '\n'.join(['%d %s' % (tid, json.dumps(el))
                             for (tid, el) in zip(ascending_indices,specs)])
        log_file.write("\n"+log_str if append else log_str)
        log_file.close()

    def __init__(self, log_path, tid_key='tid', **params):

        log_items = sorted(Log.extract_log(log_path).items())

        if tid_key is None:
            log_specs = [spec for (_, spec) in log_items]
        else:
            log_specs = [dict(list(spec.items())+[(tid_key,idx)])
                         for (idx, spec) in log_items]

        super(Log, self).__init__(log_specs,
                                  log_path=log_path,
                                  tid_key=tid_key,
                                  **params)
        self.pprint_args(['log_path'], ['tid_key'])



class FilePattern(Args):
    """
    A FilePattern specifier allows files to be matched and information
    encoded in filenames to be extracted via an extended form of
    globbing. This object may be used to specify filename arguments to
    CommandTemplates when launching jobs but it also very useful for
    collating files for analysis.

    For instance, you can find the absolute filenames of all npz files
    in a 'data' subdirectory (relative to the root) that start with
    'timeseries' using the pattern 'data/timeseries*.npz'.

    In addition to globbing supported by the glob module, patterns can
    extract metadata encoded in filenames using a subset of the Python
    format specification syntax. To illustrate, you can use
    'data/timeseries-{date}.npz' to record the date strings associated
    with matched files. Note that a particular named fields can only
    be used in a particular pattern once.

    By default metadata is extracted as strings but format types are
    supported in the usual manner
    eg. 'data/timeseries-{day:d}-{month:d}.npz' will extract the day
    and month from the filename as integer values. Only field names
    and types are recognised with other format specification syntax
    ignored. Type codes supported: 'd', 'b', 'o', 'x', 'e','E','f',
    'F','g', 'G', 'n' (if ommited, result is a string by default).

    Note that ordering is determined via ascending alphanumeric sort
    and that actual filenames should not include any globbing
    characters, namely: '?','*','[' and ']' (general good practice for
    filenames anyway).
    """

    key = param.String(default=None, allow_None=True, constant=True,
       doc='''The key name given to the matched file path strings.''')

    pattern = param.String(default=None, allow_None=True, constant=True,
       doc='''The pattern files are to be searched against.''')

    root = param.String(default=None, allow_None=True, constant=True,
       doc='''The root directory from which patterns are to be loaded.
       The root is set relative to os.getcwd().''')

    @classmethod
    def directory(cls, directory, root=None, extension=None, **kwargs):
        """
        Load all the files in a given directory selecting only files
        with the given extension if specified. The given kwargs are
        passed through to the normal constructor.
        """
        root = os.getcwd() if root is None else root
        suffix = '' if extension is None else '.' + extension.rsplit('.')[-1]
        pattern = directory + os.sep + '*' + suffix
        key = os.path.join(root, directory,'*').rsplit(os.sep)[-2]
        format_parse = list(string.Formatter().parse(key))
        if not all([el is None for el in zip(*format_parse)[1]]):
            raise Exception('Directory cannot contain format field specifications')
        return cls(key, pattern, root, **kwargs)

    def __init__(self, key, pattern, root=None, **params):
        root = os.getcwd() if root is None else root
        specs = self._load_expansion(key, root, pattern)
        super(FilePattern, self).__init__(specs, key=key, pattern=pattern,
                                          root=root, **params)
        self.pprint_args(['key', 'pattern'], ['root'])

    def fields(self):
        """
        Return the fields specified in the pattern using Python's
        formatting mini-language.
        """
        parse = list(string.Formatter().parse(self.pattern))
        return [f for f in zip(*parse)[1] if f is not None]

    def _load_expansion(self, key, root, pattern):
        """
        Loads the files that match the given pattern.
        """
        path_pattern = os.path.join(root, pattern)
        expanded_paths = self._expand_pattern(path_pattern)

        specs=[]
        for (path, tags) in expanded_paths:
            filelist = [os.path.join(path,f) for f in os.listdir(path)] if os.path.isdir(path) else [path]
            for filepath in filelist:
                specs.append(dict(tags,**{key:os.path.abspath(filepath)}))

        return sorted(specs, key=lambda s: s[key])

    def _expand_pattern(self, pattern):
        """
        From the pattern decomposition, finds the absolute paths
        matching the pattern.
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
        Given a path pattern with format declaration, generates a
        four-tuple (glob_pattern, regexp pattern, fields, type map)
        """
        sep = '~lancet~sep~'
        float_codes = ['e','E','f', 'F','g', 'G', 'n']
        typecodes = dict([(k,float) for k in float_codes]
                         + [('b',bin), ('d',int), ('o',oct), ('x',hex)])
        parse = list(string.Formatter().parse(pattern))
        text, fields, codes, _ = zip(*parse)

        # Finding the field types from format string
        types = []
        for (field, code) in zip(fields, codes):
            if code in ['', None]: continue
            constructor =  typecodes.get(code[-1], None)
            if constructor: types += [(field, constructor)]

        stars =  ['' if not f else '*' for f in fields]
        globpat = ''.join(text+star for (text,star) in zip(text,stars))

        refields = ['' if not f else sep+('(?P<%s>.*?)'% f)+sep for f in fields]
        parts = ''.join(text+group for (text,group) in zip(text, refields)).split(sep)
        for i in range(0, len(parts), 2): parts[i] = re.escape(parts[i])

        regexp_pattern = ''.join(parts).replace('\\*','.*')
        fields = list(f for f in fields if f)
        return globpat, regexp_pattern , fields, dict(types)


# Importing from filetypes requires PrettyPrinted to be defined first
from lancet.filetypes import FileType

class FileInfo(Args):
    """
    Loads metadata from a set of filenames. For instance, you can load
    metadata associated with a series of image files given by a
    FilePattern. Unlike other explicit instances of Args, this object
    extends the values of an existing Args object. Once you have
    loaded the metadata, FileInfo allows you to load the file data
    into a pandas DataFrame or a HoloViews Table.
    """

    source = param.ClassSelector(class_ = Args, doc='''
        The argument specifier that supplies the file paths.''')

    filetype = param.ClassSelector(constant=True, class_= FileType, doc='''
        A FileType object to be applied to each file path.''')

    key = param.String(constant=True, doc='''
       The key used to find the file paths for inspection.''')

    ignore = param.List(default=[], constant=True, doc='''
       Metadata keys that are to be explicitly ignored. ''')

    def __init__(self, source, key, filetype, ignore = [], **params):
        specs = self._info(source, key, filetype, ignore)
        super(FileInfo, self).__init__(specs,
                                       source = source,
                                       filetype = filetype,
                                       key = key,
                                       ignore=ignore,
                                       **params)
        self.pprint_args(['source', 'key', 'filetype'], ['ignore'])


    @classmethod
    def from_pattern(cls, pattern, filetype=None, key='filename', root=None, ignore=[]):
        """
        Convenience method to directly chain a pattern processed by
        FilePattern into a FileInfo instance.

        Note that if a default filetype has been set on FileInfo, the
        filetype argument may be omitted.
        """
        filepattern = FilePattern(key, pattern, root=root)
        if FileInfo.filetype and filetype is None:
            filetype = FileInfo.filetype
        elif filetype is None:
            raise Exception("The filetype argument must be supplied unless "
                            "an appropriate default has been specified as "
                            "FileInfo.filetype")
        return FileInfo(filepattern, key, filetype, ignore=ignore)


    @property
    def table(self):
        """
        Return an ndmapping of the loaded data using the filenames as
        values and the remaining data as the keys.
        """
        all_dimension_labels = self.constant_keys + self.varying_keys
        dimension_labels = [d for d in all_dimension_labels if d != self.key]

        if dimension_labels == []:
            return Table([spec[self.key] for spec in self.specs],
                         value_dimensions=[self.key])

        table = Table(key_dimensions=dimension_labels,
                      value_dimensions=[self.key])
        keys = []
        for spec in self.specs:
            value = spec[self.key]
            key = tuple([spec[k] for k in dimension_labels])
            if key in keys:
                key_fmt = ', '.join('%s=%r' % (k,v) for (k,v) in zip(dimension_labels, key))
                self.warning('Key clash got %s (overriding)' % key_fmt)
            table[key] = value
            keys.append(key)
        return table


    def load(self, val, **kwargs):
        """
        Load the file contents into the supplied pandas dataframe or
        HoloViews Table. This allows a selection to be made over the
        metadata before loading the file contents (may be slow).
        """
        if Table and isinstance(val, Table):
            return self.load_table(val, **kwargs)
        elif DataFrame and isinstance(val, DataFrame):
            return self.load_dframe(val, **kwargs)
        else:
            raise Exception("Type %s not a DataFrame or Table." % type(val))


    def load_table(self, table):
        """
        Load the file contents into the supplied Table using the
        specified key and filetype. The input table should have the
        filenames as values which will be replaced by the loaded
        data. If data_key is specified, this key will be used to index
        the loaded data to retrive the specified item.
        """
        items,  data_keys = [], None
        for key, filename in table.items():
            data_dict = self.filetype.data(filename[0])
            current_keys = tuple(sorted(data_dict.keys()))
            values = [data_dict[k] for k in current_keys]
            if data_keys is None:
                data_keys = current_keys
            elif data_keys != current_keys:
                raise Exception("Data keys are inconsistent")
            items.append((key, values))

        return Table(items, key_dimensions=table.key_dimensions,
                     value_dimensions=data_keys)


    def load_dframe(self, dframe):
        """
        Load the file contents into the supplied dataframe using the
        specified key and filetype.
        """
        filename_series = dframe[self.key]
        loaded_data = filename_series.map(self.filetype.data)
        keys = [list(el.keys()) for el in loaded_data.values]
        for key in set().union(*keys):
            key_exists = key in dframe.columns
            if key_exists:
                self.warning("Appending '_data' suffix to data key %r to avoid"
                             "overwriting existing metadata with the same name." % key)
            suffix = '_data' if key_exists else ''
            dframe[key+suffix] = loaded_data.map(lambda x: x.get(key, np.nan))
        return dframe


    def _info(self, source, key, filetype, ignore):
        """
        Generates the union of the source.specs and the metadata
        dictionary loaded by the filetype object.
        """
        specs, mdata = [], {}
        mdata_clashes  = set()

        for spec in source.specs:
            if key not in spec:
                raise Exception("Key %r not available in 'source'." % key)

            mdata = dict((k,v) for (k,v) in filetype.metadata(spec[key]).items()
                         if k not in ignore)
            mdata_spec = dict(spec, **mdata)
            specs.append(mdata_spec)
            mdata_clashes = mdata_clashes | (set(spec.keys()) & set(mdata.keys()))
        # Metadata clashes can be avoided by using the ignore list.
        if mdata_clashes:
            self.warning("Loaded metadata keys overriding source keys.")
        return specs
