#
# Lancet IPython support
# Load with: %load_ext lancet.ipython
#

import os

import lancet.core as core
import lancet.launch as launch

def StaticArgs_repr_html(obj, cols=None, max_rows=None):
    """
    Generate a HTML table for the specifier.
    """
    def _html_row(spec, columns):
        row_strings = []
        for value in [spec[col] for col in columns]:
            html_repr = str
            html_formatters = get_ipython().display_formatter.formatters['text/html']
            for (cls,format_fn) in html_formatters.type_printers.items():
                if isinstance(value, cls):
                    html_repr = format_fn
                    break
            row_strings.append('<td>'+html_repr(value)+'</td>')
        return ' '.join(['<tr>'] + row_strings + ['</tr>'])

    max_rows = len(obj) if max_rows is None else max_rows
    columns = obj.varying_keys() if cols is None else cols

    all_varying = obj.varying_keys()
    if not all(col in all_varying for col in columns):
        raise Exception('Columns must belong to the varying keys')

    summary = '<tr><td><b>%r<br>[%d items]</b></td></tr>' % (obj.__class__.__name__, len(obj))
    cspecs = [{'Key':k, 'Value':v} for (k,v) in obj.constant_items()]
    crows = [_html_row(spec, ['Key', 'Value']) for spec in cspecs]
    cheader_str = '<tr><td><b>Constant Key</b></td><td><b>Value</b></td></tr>'

    vrows = [_html_row(spec,columns) for spec in obj.specs[:max_rows]]
    vheader_str= ' '.join(['<tr>'] + ['<td><b>'+str(col)+'</b></td>' for col in columns ] +['</tr>'])
    ellipses = ' '.join(['<tr>'] + ['<td>...</td>' for col in columns ] +['</tr>'])
    ellipse_str = ellipses  if (max_rows < len(obj)) else ''

    html_elements = ['<table>', summary, cheader_str] + crows + [vheader_str] + vrows + [ellipse_str, '</table>']
    html = '\n'.join(html_elements)
    return html

def StaticArgs_repr_pretty(obj, p, cycle):
    p.text(obj._pprint(cycle, annotate=True))

def applying_repr_pretty(obj, p, cycle):
    annotation = ('# == %d items accumulated, callee=%r ==\n' %
                  (len(obj.accumulator),
                   obj.callee.__name__ if hasattr(obj.callee, '__name__') else 'None'))
    p.text(annotation + str(obj))

class review_and_launch:
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
            prelude = ['from lancet import *',
                       '%load_ext lancet.ipython']
            header_str =  '\n'.join(header) + ccell + ccell.join(prelude)

            html_reprs = [ccell+'(%r)' % lval[0].arg_specifier for lval in lvals]
            zipped = [(mcell+'# #### Launch %d' %i, r) for (i,r) in enumerate(html_reprs)]
            body_str = ''.join([val for pair in zipped for val in pair])
            node = current.reads(header_str + body_str, 'py')
            current.write(node, open(nb_path, 'w'), 'ipynb')
            print("Saved to %s " % nb_path)

    def _repr_pretty_(self, p, cycle):
        annotation = ('# == launch_fn=%r ==\n' %
                      (self.launch_fn.__name__ if hasattr(self.launch_fn, '__name__') else 'None',))
        p.text(annotation+ str(self))

_loaded = False

def load_ipython_extension(ip):
    global _loaded

    if not _loaded:
        _loaded = True

        plaintext_formatter = ip.display_formatter.formatters['text/plain']
        html_formatter = ip.display_formatter.formatters['text/html']

        plaintext_formatter.for_type(core.StaticArgs, StaticArgs_repr_pretty)
        html_formatter.for_type(core.StaticArgs, StaticArgs_repr_html)

        plaintext_formatter.for_type(core.applying, applying_repr_pretty)
        plaintext_formatter.for_type(launch.review_and_launch, review_and_launch._repr_pretty_)

        launch.review_and_launch._process_launchers = review_and_launch._process_launchers

