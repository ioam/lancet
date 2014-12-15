#
# Work in progress. Only SimpleOptimization is currently available.
#

import os, json, fnmatch
import param
from lancet.core import Arguments, Concatenate, CartesianProduct

class DynamicArgs(Arguments):
    """
    DynamicArgs are declarative specifications that specify a
    parameter space via a dynamic algorithmic process instead of using
    precomputed arguments. Unlike the static Args objects, new
    arguments can only be generated in response to some feedback from
    the processes that are executed. This type of dynamic feedback is
    a common feature for many algorithms such as hill climbing
    optimization, genetic algorithms, bisection search and other
    sophisticated optimization and search procedures.

    Like the Args objects, a DynamicArgs object is an iterator. On
    each iteration, one or more argument sets defining a collection of
    independent jobs are returned (these jobs should be possible to
    execute concurrently). Between iterations, the output_extractor
    function is used to extract the necessary information from the
    standard output streams of the previously executed jobs. This
    information is used to update the internal state of the
    DynamicArgs object can then generate more arguments to explore or
    terminate. All DynamicArgs classes need to declare the expected
    return format of the output_extractor function.
    """

    output_extractor = param.Callable(default=json.loads, doc="""
       The function that returns the relevant data from the standard
       output stream dumped to file in the streams subdirectory. This
       information must be retyrned in a format suitable for updating
       the specifier.. By default uses json.loads but pickle.loads
       could also be a valid option. The callable must take a string
       and return a Python object suitable for updating the specifier.""")


    def __init__(self, **params):
        super(DynamicArgs, self).__init__(**params)
        # Returned on next iteration and updated by _updated_state method
        self._next_val = self._initial_state(**params)
        # Trace of inputs from output_extractor and returned arguments
        self.trace = [(None, self._next_val)]

    def _update_state(self, data):
        """
        Determines the next desired point in the parameter space using
        the parsed data returned from the public update
        method. Returns the value that will next emitted on the next
        iteration using the list of dictionaries format for
        arguments. If the update fails or data is None, StopIteration
        should be supplied as the return value.
        """
        raise NotImplementedError


    def _initial_state(self, **kwargs):
        """
        Reset the the DynamicArgs object to its initial state and used
        to reset the object adter the iterator is exhausted. The
        return value is the initial argument to be returned by
        next().
        """
        raise NotImplementedError

    def __next__(self):
        if self._next_val is StopIteration:
            self._initial_state()
            raise StopIteration
        current_val =  self._next_val
        self._next_val = StopIteration
        return current_val

    next = __next__


    def update(self, tids, info):
        """
        Called to update the state of the iterator.  This methods
        receives the set of task ids from the previous set of tasks
        together with the launch information to allow the output
        values to be parsed using the output_extractor. This data is then
        used to determine the next desired point in the parameter
        space by calling the _update_state method.
        """
        outputs_dir = os.path.join(info['root_directory'], 'streams')
        pattern = '%s_*_tid_*{tid}.o.{tid}*' % info['batch_name']
        flist = os.listdir(outputs_dir)
        try:
            outputs = []
            for tid in tids:
                matches = fnmatch.filter(flist, pattern.format(tid=tid))
                if len(matches) != 1:
                    self.warning("No unique output file for tid %d" % tid)
                contents = open(os.path.join(outputs_dir, matches[0]),'r').read()
                outputs.append(self.output_extractor(contents))

            self._next_val = self._update_state(outputs)
            self.trace.append((outputs, self._next_val))
        except:
            self.warning("Cannot load required output files. Cannot continue.")
            self._next_val = StopIteration


    def show(self):
        """
        When dynamic, not all argument values may be available.
        """
        copied = self.copy()
        enumerated = [el for el in enumerate(copied)]
        for (group_ind, specs) in enumerated:
            if len(enumerated) > 1: print("Group %d" % group_ind)
            ordering = self.constant_keys + self.varying_keys
            # Ordered nicely by varying_keys definition.
            spec_lines = [', '.join(['%s=%s' % (k, s[k]) for k in ordering]) for s in specs]
            print('\n'.join(['%d: %s' % (i,l) for (i,l) in enumerate(spec_lines)]))

        print('Remaining arguments not available for %s' % self.__class__.__name__)


    def _trace_summary(self):
        """
        Summarizes the trace of values used to update the DynamicArgs
        and the arguments subsequently returned. May be used to
        implement the summary method.
        """
        for (i, (val, args)) in enumerate(self.trace):
            if args is StopIteration:
                info = "Terminated"
            else:
                pprint = ','.join('{' + ','.join('%s=%r' % (k,v)
                         for (k,v) in arg.items()) + '}' for arg in args)
                info = ("exploring arguments [%s]" % pprint )

            if i == 0: print("Step %d: Initially %s." % (i, info))
            else:      print("Step %d: %s after receiving input(s) %s." % (i, info.capitalize(), val))

    def __add__(self, other):
        """
        Concatenates two argument specifiers. See Concatenate and
        DynamicConcatenate documentation respectively.
        """
        if not other: return self
        dynamic = (isinstance(self, DynamicArgs),  isinstance(other, DynamicArgs))
        if dynamic == (True, True):
            raise Exception('Cannot concatenate two dynamic specifiers.')
        elif (True in dynamic):
            return DynamicConcatenate(self,other)
        else:
            return Concatenate(self,other)


    def __mul__(self, other):
        """
        Takes the cartesian product of two argument specifiers. See
        CartesianProduct and DynamicCartesianProduct documentation.
        """
        if not other: return []
        dynamic = (isinstance(self, DynamicArgs),  isinstance(other, DynamicArgs))
        if dynamic == (True, True):
            raise Exception('Cannot take Cartesian product two dynamic specifiers.')
        elif (True in dynamic):
            return DynamicCartesianProduct(self, other)
        else:
            return CartesianProduct(self, other)


    def __len__(self):
        """
        Many DynamicArgs won't have a length that can be
        precomputed. Most DynamicArgs objects will have an iteration
        limit to guarantee eventual termination. If so, the maximum
        possible number of arguments that could be generated should be
        returned.
        """
        raise NotImplementedError



class SimpleGradientDescent(DynamicArgs):
    """
    Very simple gradient descent optimizer designed to illustrate how
    Dynamic Args may be implemented. This class has been deliberately
    kept simple to clearly illustrate how Dynamic Args may be
    implemented. A more practical example would likely probably make
    use of mature, third party optimization libraries (such as the
    routines offered in scipy.optimize).

    This particular algorithm greedily minimizes an output value via
    greedy gradient descent. The local parameter space is explored by
    examining the change in output value when an increment or
    decrement of 'stepsize' is made in the parameter space, centered
    around the current position. The initial parameter is initialized
    with the 'start' value and the optimization process terminates
    when either a local minima/maxima has been found or when
    'max_steps' is reached.

    The 'output_extractor' function is expected to return a single
    scalar number to drive the gradient descent algorithm forwards.
    """

    key = param.String(constant=True, doc="""
        The name of the argument that will be optimized in a greedy fashion.""")

    start = param.Number(default=0.0, constant=True, doc="""
        The starting argument value for the gradient ascent or descent""")

    stepsize = param.Number(default=1.0, constant=True, doc="""
        The size of the steps taken in parameter space.""")

    max_steps=param.Integer(default=100, constant=True, doc="""
        Once max_steps is reached, the optimization terminates.""")

    def __init__(self, key, **params):
        super(SimpleGradientDescent, self).__init__(key=key, **params)
        self.pprint_args(['key', 'start', 'stepsize'],[])
        self._termination_info = None

    def _initial_state(self, **kwargs):
        self._steps_complete = 0
        self._best_val = float('inf')
        self._arg = self.start
        return [{self.key:self.start+1}, {self.key:self.start-1}]

    def _update_state(self, vals):
        """
        Takes as input a list or tuple of two elements. First the
        value returned by incrementing by 'stepsize' followed by the
        value returned after a 'stepsize' decrement.
        """
        self._steps_complete += 1
        if self._steps_complete == self.max_steps:
            self._termination_info = (False, self._best_val, self._arg)
            return StopIteration

        arg_inc, arg_dec = vals
        best_val = min(arg_inc, arg_dec, self._best_val)
        if best_val == self._best_val:
            self._termination_info = (True, best_val, self._arg)
            return StopIteration

        self._arg += self.stepsize if (arg_dec > arg_inc) else -self.stepsize
        self._best_val= best_val
        return [{self.key:self._arg+self.stepsize},
                {self.key:self._arg-self.stepsize}]

    @property
    def constant_keys(self):  return []
    @property
    def constant_items(self): return []
    @property
    def varying_keys(self):   return [self.key]

    def summary(self):
        print('Varying Keys: %r' % self.key)
        print('Maximum steps allowed: %d' % self.max_steps)
        self._trace_summary()
        (val, arg) = (self.trace[-1])
        if self._termination_info:
            (success, best_val, arg) = self._termination_info
            condition =  'Successfully converged.' if success else 'Maximum step limit reached.'
            print("%s Minimum value of %r at %s=%r." % (condition, best_val, self.key, arg))

    def __len__(self):
        return 2*self.max_steps # Each step specifies 2 concurrent jobs



#=========================#
# Experimental code (WIP) #
#=========================#


class DynamicConcatenate(DynamicArgs):
    def __init__(self, first, second):
        self.first = first
        self.second = second
        super(Concatenate, self).__init__(dynamic=True)

        self._exhausted = False
        self._first_sent = False
        self._first_cached = None
        self._second_cached = None
        if not isinstance(first,DynamicArgs):
            self._first_cached = next(first.copy())
        if not isinstance(second,DynamicArgs):
            self._second_cached = next(second.copy())
        self.pprint_args(['first', 'second'],[], infix_operator='+')


    def constant_keys(self):
        return list(set(self.first.constant_keys) | set(self.second.constant_keys))


    def varying_keys(self):
        return list(set(self.first.varying_keys) | set(self.second.varying_keys))


    def update(self, tids, info):
        if (self.isinstance(self.first,DynamicArgs) and not self._exhausted):
            self.first.update(tids, info)
        elif (self.isinstance(self.second,DynamicArgs) and self._first_sent):
            self.second.update(tids, info)

    def __next__(self):
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

    next = __next__


class DynamicCartesianProduct(DynamicArgs):

    def __init__(self, first, second):

        self.first = first
        self.second = second

        overlap = set(self.first.varying_keys) &  set(self.second.varying_keys)
        assert overlap == set(), 'Sets of keys cannot overlap between argument specifiers in cartesian product.'

        super(CartesianProduct, self).__init__(dynamic=True)

        self._first_cached = None
        self._second_cached = None
        if not isinstance(first,DynamicArgs):
            self._first_cached = next(first.copy())
        if not isinstance(second,DynamicArgs):
            self._second_cached = next(second.copy())

        self.pprint_args(['first', 'second'],[], infix_operator='*')

    def constant_keys(self):
        return list(set(self.first.constant_keys) | set(self.second.constant_keys))


    def varying_keys(self):
        return list(set(self.first.varying_keys) | set(self.second.varying_keys))


    def update(self, tids, info):
        if self.isinstance(self.first,DynamicArgs):  self.first.update(tids, info)
        if self.isinstance(self.second,DynamicArgs): self.second.update(tids, info)


    def __next__(self):
        if self._first_cached is None:
            first_spec = next(self.first)
            return self._cartesian_product(first_spec, self._second_cached)
        else:
            second_spec = next(self.second)
            return self._cartesian_product(self._first_cached, second_spec)
    next = __next__
