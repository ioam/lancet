#
# Experimental code (WIP)
#

import os, sys, json, time, fnmatch, pipes, subprocess
import param
from core import BaseArgs, Concatenate, CartesianProduct

class DynamicArgs(BaseArgs):

    output_extractor = param.Callable(default=json.loads, doc="""
       The function that returns the relevant data from the standard
       output stream dumped to file in the streams subdirectory. This
       information must be retyrned in a format suitable for updating
       the specifier.. By default uses json.loads but pickle.loads
       could also be a valid option. The callable must take a string
       and return a python object suitable for updating the specifier.""")


    def __init__(self, **kwargs):
        super(DynamicArgs, self).__init__(**kwargs)
        # Returned next iteration and updated by _updated_state method
        self._next_val = self._initial_state(**kwargs)


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
        return value is the initial argument to be returned by next().
        """
        raise NotImplementedError


    def next(self):
        if self._next_val is StopIteration:
            self._initial_state()
            raise StopIteration
        current_val =  self._next_val
        self._next_val = StopIteration
        return current_val


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
        except:
            self.warning("Cannot load required output files. Cannot continue.")
            return None # StopIteration should be raised by the argument specifier


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


#=============================#
# Dynamic argument specifiers #
#=============================#


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


    def next(self):
        if self._first_cached is None:
            first_spec = next(self.first)
            return self._cartesian_product(first_spec, self._second_cached)
        else:
            second_spec = next(self.second)
            return self._cartesian_product(self._first_cached, second_spec)



from launch import CommandTemplate

class DynamicCommandTemplate(CommandTemplate):
    """
    Command templates should avoid all platform specific logic (this
    is the job of the Launcher) but there are cases where the
    specification needs to be read from file. This allows tasks to be
    queued (typically on a cluster) before the required arguments are
    known. To achieve this two extra methods should be implemented if
    possible:

    specify(spec, tid, info):

    Takes a specification for the task and writes it to a file in the
    'specifications' subdirectory of the root directory. Each
    specification filename must include the tid to ensure uniqueness.

    queue(tid, info):

    The subprocess Popen argument list that upon execution runs the
    desired command using the arguments in the specification file of
    the given tid located in the 'specifications' folder of the root
    directory.
    """

    def specify(self,spec, tid, info):
        raise NotImplementedError


    def queue(self,tid, info):
        raise NotImplementedError


    def show(self, args, file_handle=sys.stdout, queue_cmd_only=False):
        info = {'root_directory':     '<root_directory>',
                'batch_name':         '<batch_name>',
                'batch_tag':          '<batch_tag>',
                'batch_description':  '<batch_description>',
                'timestamp':          tuple(time.localtime()),
                'varying_keys':       args.varying_keys,
                'constant_keys':      args.constant_keys,
                'constant_items':     args.constant_items}

        if queue_cmd_only and not hasattr(self, 'queue'):
            print("Cannot show queue: CommandTemplate does not allow queueing")
            return
        elif queue_cmd_only:
            full_string = 'Queue command: '+ ' '.join([pipes.quote(el) for el in self.queue('<tid>',info)])
        elif (queue_cmd_only is False):
            copied = args.copy()
            full_string = ''
            enumerated = list(enumerate(copied))
            for (group_ind, specs) in enumerated:
                if len(enumerated) > 1: full_string += "\nGroup %d" % group_ind
                quoted_cmds = [ subprocess.list2cmdline(
                        [el for el in self(self._formatter(copied, s),'<tid>',info)])
                                for s in specs]
                cmd_lines = ['%d: %s\n' % (i, qcmds) for (i,qcmds) in enumerate(quoted_cmds)]
                full_string += ''.join(cmd_lines)

        file_handle.write(full_string)
        file_handle.flush()
