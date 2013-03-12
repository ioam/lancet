#
# Lancet IPython support
#

import os, sys
import IPython
from IPython.display import HTML
import param
import lancet.core as core

# Provide default versions from core
from lancet.core import set_fp_precision, BaseArgs, CommandTemplate, \
        Launcher, QLauncher

import lancet
lancet._module = sys.modules[__name__]

class IPythonHTMLArgs(param.Parameterized):
    """
    Mixin class for IPython Notebook support; provides html output.
    """
    def _html_row(self, spec, columns):
        row_strings = []
        for value in [spec[col] for col in columns]:
            html_repr = value.html(html_fn=str) if hasattr(value, 'html') else str(value)
            row_strings.append('<td>'+html_repr+'</td>')
        return ' '.join(['<tr>'] + row_strings + ['</tr>'])

    def html(self, cols=None, html_fn=None, max_rows=None):
        """
        Generate a HTML table for the specifier.
        """
        html_fn = HTML if html_fn is None else html_fn
        max_rows = len(self) if max_rows is None else max_rows
        columns = self.varying_keys() if cols is None else cols

        all_varying = self.varying_keys()
        if not all(col in all_varying for col in columns):
            raise Exception('Columns must belong to the varying keys')

        summary = '<tr><td><b>%r<br>[%d items]</b></td></tr>' % (self.__class__.__name__, len(self))
        cspecs = [{'Key':k, 'Value':v} for (k,v) in self.constant_items()]
        crows = [self._html_row(spec, ['Key', 'Value']) for spec in cspecs]
        cheader_str = '<tr><td><b>Constant Key</b></td><td><b>Value</b></td></tr>'

        vrows = [self._html_row(spec,columns) for spec in self.specs[:max_rows]]
        vheader_str= ' '.join(['<tr>'] + ['<td><b>'+str(col)+'</b></td>' for col in columns ] +['</tr>'])
        ellipses = ' '.join(['<tr>'] + ['<td>...</td>' for col in columns ] +['</tr>'])
        ellipse_str = ellipses  if (max_rows < len(self)) else ''

        html_elements = ['<table>', summary, cheader_str] + crows + [vheader_str] + vrows + [ellipse_str, '</table>']
        html = '\n'.join(html_elements)
        return html_fn(html)

class StaticArgs(core.StaticArgs, IPythonHTMLArgs):
    def _repr_pretty_(self, p, cycle): p.text(self._pprint(cycle, annotate=True))

class StaticConcatenate(core.StaticConcatenate, IPythonHTMLArgs):
    def _repr_pretty_(self, p, cycle): p.text(self._pprint(cycle, annotate=True))

class StaticCartesianProduct(core.StaticCartesianProduct, IPythonHTMLArgs):
    def _repr_pretty_(self, p, cycle): p.text(self._pprint(cycle, annotate=True))

class Args(core.Args, IPythonHTMLArgs):
    def _repr_pretty_(self, p, cycle): p.text(self._pprint(cycle, annotate=True))

class LinearArgs(core.LinearArgs, IPythonHTMLArgs):
    def _repr_pretty_(self, p, cycle): p.text(self._pprint(cycle, annotate=True))

class ListArgs(core.ListArgs, IPythonHTMLArgs):
    def _repr_pretty_(self, p, cycle): p.text(self._pprint(cycle, annotate=True))

class Log(core.Log, IPythonHTMLArgs):
    def _repr_pretty_(self, p, cycle): p.text(self._pprint(cycle, annotate=True))

class Indexed(core.Indexed, IPythonHTMLArgs):
    def _repr_pretty_(self, p, cycle): p.text(self._pprint(cycle, annotate=True))

class FilePattern(core.FilePattern, IPythonHTMLArgs):
    def _repr_pretty_(self, p, cycle): p.text(self._pprint(cycle, annotate=True))

class LexSorted(core.LexSorted, IPythonHTMLArgs):
    def _repr_pretty_(self, p, cycle): p.text(self._pprint(cycle, annotate=True))

class DynamicConcatenate(core.DynamicConcatenate):
    def _repr_pretty_(self, p, cycle): p.text(self._pprint(cycle, annotate=True))

class DynamicCartesianProduct(core.DynamicCartesianProduct):
    def _repr_pretty_(self, p, cycle): p.text(self._pprint(cycle, annotate=True))

class applying(core.applying):
    def _repr_pretty_(self, p, cycle):
        annotation = ('# == %d items accumulated, callee=%r ==\n' %
                      (len(self.accumulator),
                       self.callee.__name__ if hasattr(self.callee, '__name__') else 'None'))
        p.text(annotation+ str(self))

class review_and_launch(core.review_and_launch):
    def _process_launchers(self, lvals):
        """
        Saves the launch specifiers together in an IPython notebook for
        convenient viewing. Only offered as an option if IPython is available.
        """

        from IPython.nbformat import current
        notebook_dir = os.environ.get('LANCET_NB_DIR',None)
        notebook_dir = notebook_dir if notebook_dir else os.getcwd()

        if self.input_options(['y','N'], 'Save IPython notebook?', default='n') == 'y':
            print('Notebook directory ($LANCET_NB_DIR): %s' % notebook_dir)
            isdir = False
            while not isdir:
                fname = raw_input('Filename: ').replace(' ','_')
                fname = fname if fname.endswith('.ipynb') else fname+'.ipynb'
                nb_path = os.path.abspath(os.path.join(notebook_dir, fname))
                isdir = os.path.isdir(os.path.split(nb_path)[0])
                if not isdir:  print('Invalid directory %s' % os.path.split(nb_path)[0])

            ccell = '\n# <codecell>\n'; mcell='\n# <markdowncell>\n'
            header = ['# -*- coding: utf-8 -*-','# <nbformat>3.0</nbformat>']
            prelude = ['from lancet import *']
            header_str =  '\n'.join(header) + ccell + ccell.join(prelude)

            html_reprs = [ccell+'(%r).html()' % lval[0].arg_specifier for lval in lvals]
            zipped = [(mcell+'# #### Launch %d' %i, r) for (i,r) in enumerate(html_reprs)]
            body_str = ''.join([val for pair in zipped for val in pair])
            node = current.reads(header_str + body_str, 'py')
            current.write(node, open(nb_path, 'w'), 'ipynb')
            print("Saved to %s " % nb_path)

    def _repr_pretty_(self, p, cycle):
        annotation = ('# == launch_fn=%r ==\n' %
                      (self.launch_fn.__name__ if hasattr(self.launch_fn, '__name__') else 'None',))
        p.text(annotation+ str(self))

