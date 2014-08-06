import os, tempfile, json, pickle
import param
from lancet.core import PrettyPrinted

try:
    from io import StringIO
except:
    from StringIO import StringIO

try:    import numpy
except: pass

try:    import Image
except: pass

class FileType(PrettyPrinted, param.Parameterized):
    """
    The base class for all supported file types in Lancet. This class
    is designed to be simple and easily extensible to support new
    files and has only three essential methods: 'save', 'data' and
    'metadata').

    Optionally, a 'display' classmethod may be implemented to
    allow the file contents to be quickly visualized.
    """

    hash_suffix = param.Boolean(default=True, doc='''
       Whether to ensure the saved filename is unique by adding a
       short hash suffix. Note that this is a class level parameter
       only.''')

    directory = param.String(default='.', allow_None=True, doc='''
       Directory in which to load or save the file. Note that this
       is a class level parameter only.''')

    extensions = param.List(default=[], constant=True,
       doc= '''The set of supported file extensions.''')

    data_key = param.String(default='data', doc='''
       The name (key) given to the file contents if the key cannot be
       determined from the file itself.''')

    def __init__(self, **kwargs):
        super(FileType, self).__init__(**kwargs)
        self._pprint_args = ([],[],None,{})
        self.pprint_args(['data_key', 'hash_suffix'], [])

    def save(self, filename,  metadata={}, **data):
        """
        The implementation in the base class simply checks there is no
        clash between the metadata and data keys.
        """
        intersection = set(metadata.keys()) & set(data.keys())
        if intersection:
            msg = 'Key(s) overlap between data and metadata: %s'
            raise Exception(msg  % ','.join(intersection))

    def metadata(self, filename):
        """
        The metadata returned as a dictionary.
        """
        raise NotImplementedError

    def data(self, filename):
        """
        Data returned as a dictionary.
        """
        raise NotImplementedError

    def _loadpath(self, filename):
        (_, ext) = os.path.splitext(filename)
        if ext not in self.extensions:
            raise Exception("Unsupported extensions")
        abspath = os.path.abspath(os.path.join(self.directory, filename))
        return filename if os.path.isfile(filename) else abspath

    def _savepath(self, filename):
        """
        Returns the full path for saving the file, adding an extension
        and making the filename unique as necessary.
        """
        (basename, ext) = os.path.splitext(filename)
        ext = ext if ext else self.extensions[0]
        savepath = os.path.abspath(os.path.join(self.directory,
                                                 '%s%s' % (basename, ext)))
        return (tempfile.mkstemp(ext, basename + "_", self.directory)[1]
                if self.hash_suffix else savepath)

    @classmethod
    def file_supported(cls, filename):
        """
        Returns a boolean indicating whether the filename has an
        appropriate extension for this class.
        """
        if not isinstance(filename, str):
            return False
        (_, ext) = os.path.splitext(filename)
        if ext not in cls.extensions:
            return False
        else:
            return True

    @classmethod
    def _img_tag(cls, data, size, format='png'):
        """
        Helper to conviently build a base64 encoded img tag. Accepts
        both png and svg image types.  Useful for implementing the
        display method when available.
        """
        assert format in ['png', 'svg'], "Only png or svg display supported"
        prefix = ('data:image/png;base64,' if format=='png'
                  else 'data:image/svg+xml;base64,')
        b64 = prefix + data.encode("base64")
        return ("<img height='%d' width='%d' src='%s' />" % (size, size, b64))

    @classmethod
    def display(cls, value, size=64, format='png'):
        """
        A function that generates a display image as a base64 encoded
        HTMl image (png format). The input should either by a filename
        with an appropriate extension or the appropriate object
        type. Should be implemented if there is an obvious
        visualization suitable for the file type/object. If the value
        cannot be visualized with this class, return None.
        """
        return None

    def __repr__(self):
        return self._pprint(flat=True, annotate=False)

    def __str__(self):
        return self._pprint(annotate=False)

    def __or__(self, other):
        return FileOption(self, other)



class FileOption(FileType):
    """
    Allows a FileType out of a pair of FileTypes to handle a given
    file. For instance, given a mixed list of .png image filenames and
    .npz Numpy filenames, the ImageFile() | NumpyFile() object an
    handle either type of filename appropriately.
    """
    first = param.ClassSelector(class_=FileType, doc='''
       The first potential FileType to handle a given filename.''')

    second = param.ClassSelector(class_=FileType, doc='''
       The second potential FileType to handle a given filename.''')

    def __init__(self, first, second, **kwargs):
        if set(first.extensions) & set(second.extensions):
            raise Exception("FileTypes must support non-overlapping sets of extensions.")
        extensions = set(first.extensions) | set(second.extensions)
        super(FileOption, self).__init__(first=first, second=second,
                                         extensions = list(extensions), **kwargs)
        self.pprint_args(['first', 'second'],[], infix_operator='|')

    def save(self, filename,  metadata={}, **data):
        raise Exception("A FileChoice cannot be used to save data.")

    def metadata(self, filename):
        self._loadpath(filename) # Ensures a valid extension

        try:    first_metadata = self.first.metadata(filename)
        except: first_metadata = {}
        try:     second_metadata = self.second.metadata(filename)
        except:  second_metadata = {}
        return dict(first_metadata, **second_metadata)

    def data(self, filename):
        self._loadpath(filename) # Ensures a valid extension

        try:     first_data = self.first.data(filename)
        except:  first_data = {}
        try:    second_data = self.second.data(filename)
        except: second_data = {}
        return dict(first_data, **second_data)

    def __repr__(self):
        return self._pprint(flat=True, annotate=False)

    def __str__(self):
        return self._pprint(annotate=False)


class CustomFile(FileType):
    """
    A customizable FileType that takes two functions as input and maps
    them to the loading interface for all FileTypes.
    """

    data_fn = param.Callable(doc='''
       A callable that takes a filename and returns a dictionary of
       data values''')

    metadata_fn = param.Callable(doc='''
        A callable that takes a filename and returns a dictionary of
        metadata values''')

    def __init__(self, data_fn=None, metadata_fn=None, **kwargs):
        zipped = zip(['data_fn','metadata_fn'], [data_fn, metadata_fn])
        fn_dict = dict([(k,v) for (k,v) in zipped if (v is not None)])
        super(CustomFile, self).__init__(**dict(kwargs, **fn_dict))

    def save(self, filename, data):
        raise NotImplementedError

    def data(self, filename):
        val = self.data_fn(filename)
        if not isinstance(val, dict):
            val = {self.data_key:val}
        return val

    def metadata(self, filename):
        val = self.metadata_fn(filename)
        if not isinstance(val, dict):
            raise Exception("The metadata callable must return a dictionary.")
        return val



class JSONFile(FileType):
    """
    It is assumed you won't store very large volumes of data as JSON.
    For this reason, the contents of JSON files are loaded as
    metadata.
    """
    extensions = param.List(default=['.json'], constant=True)

    def __init__(self, **kwargs):
        super(JSONFile, self).__init__(**kwargs)
        self.pprint_args(['hash_suffix'], [])

    def save(self, filename, metadata={}):
        jsonfile = open(self._savepath(filename),'wb')
        json.dump(metadata, jsonfile)

    def metadata(self, filename):
        jsonfile = open(self._loadpath(filename),'r')
        jsondata = json.load(jsonfile)
        jsonfile.close()
        return jsondata

    def data(self, filename):
        raise Exception("JSONFile only loads metadata")



class NumpyFile(FileType):
    """
    An npz file is the standard way to save Numpy arrays. This is a
    highly flexible FileType that supports most native Python objects
    including Numpy arrays.
    """

    extensions = param.List(default=['.npz'], constant=True)

    compress = param.Boolean(default=True, doc="""
      Whether or not the compressed npz format should be used.""")

    def __init__(self, **kwargs):
        super(NumpyFile, self).__init__(**kwargs)
        self.pprint_args(['hash_suffix'], ['compress']) # CHECK!

    def save(self, filename, metadata={}, **data):
        super(NumpyFile, self).save(filename, metadata, **data)
        savefn = numpy.savez_compressed if self.compress else numpy.savez
        savefn(self._savepath(filename), metadata=metadata, **data)

    def metadata(self, filename):
        npzfile = numpy.load(self._loadpath(filename))
        metadata = (npzfile['metadata'].tolist()
                    if 'metadata' in list(npzfile.keys()) else {})
        # Numpy load may return a Python dictionary.
        if not isinstance(npzfile, dict): npzfile.close()
        return metadata

    def data(self, filename):
        npzfile = numpy.load(self._loadpath(filename))
        keys = [k for k in npzfile.keys() if k != 'metadata']
        data = dict((k,npzfile[k]) for k in keys)

        # Is this a safe way to unpack objects?
        for (k,val) in data.items():
            if val.dtype.char == 'O' and val.shape == ():
                data[k] = val[()]

        if not isinstance(npzfile, dict):
            npzfile.close()
        return data


class ViewFile(FileType):
    """
    An .view file contains a collection of DataViews stored on an
    AttrTree. The file itself is a NumpyFile containing a pickle of
    the AttrTree object as well as any additional metadata.
    """

    extensions = param.List(default=['.view'], constant=True)

    filters = param.List(default=[], doc="""
      A list of path tuples used to select data from the loaded
      AttrTree. The paths may specify complete path matches or just a
      partial path in the tree (which selects the corresponding
      subtree). If empty, no filtering is applied.""")

    compress = param.Boolean(default=True, doc="""
      Whether or not the compressed npz format should be used.""")

    def __init__(self, **kwargs):
        super(ViewFile, self).__init__(**kwargs)
        self.pprint_args(['hash_suffix', 'filters'], ['compress'])

    def save(self, filename, path_index, metadata={}):
        super(ViewFile, self).save(filename, metadata, data=path_index)
        savefn = numpy.savez_compressed if self.compress else numpy.savez
        filename = self._savepath(filename)
        savefn(open(filename, 'w'), metadata=metadata, data=path_index)

    def metadata(self, filename):
        npzfile = numpy.load(self._loadpath(filename))
        metadata = (npzfile['metadata'].tolist()
                    if 'metadata' in npzfile.keys() else {})
        # Numpy load may return a Python dictionary.
        if not isinstance(npzfile, dict): npzfile.close()
        return metadata


    def data(self, filename):
        from dataviews.collector import AttrTree
        npzfile = numpy.load(self._loadpath(filename))
        keys = [k for k in npzfile.keys() if k != 'metadata']
        data = dict((k,npzfile[k]) for k in keys)

        for (k,val) in data.items():
            if val.dtype.char == 'O' and val.shape == ():
                data[k] = val[()]

        if not isinstance(npzfile, dict):
            npzfile.close()

        # Filter the AttrTree using any specified filters
        data = data['data']
        filters = [(f,) if isinstance(f,str) else f for f in self.filters]
        if filters:
            path_items = set((k,v) for (k,v) in data.path_items.items()
                             for f in filters if k[:len(f)]==f)
            retdata = AttrTree()
            for path, val in path_items:
                retdata.set_path(path, val)
            data = retdata
        return {self.data_key:data}



class ImageFile(FileType):
    """
    Image support - requires PIL or Pillow.
    """
    extensions = param.List(default=['.png', '.jpg'], constant=True)

    image_info = param.Dict(default={'mode':'mode', 'size':'size', 'format':'format'},
        doc='''Dictionary of the metadata to load. Each key is the
        name given to the metadata item and each value is the PIL
        Image attribute to return.''')

    data_mode = param.ObjectSelector(default='RGBA',
                                     objects=['L', 'RGB', 'RGBA', 'I','F'],
        doc='''Sets the mode of the mode of the Image object. Palette
        mode'P is not supported''')

    data_key = param.String(default='images', doc='''
       The name (key) given to the loaded image data.''')

    def __init__(self, **kwargs):
        super(ImageFile, self).__init__(**kwargs)
        self.pprint_args(['hash_suffix'],
                         ['data_key', 'data_mode', 'image_info'])

    def metadata(self, filename, **kwargs):
        image = Image.open(self._loadpath(filename))
        return dict((name, getattr(image,attr,None))
                    for (name, attr) in self.image_info.items())

    def save(self, filename, imdata, **data):
        """
        Data may be either a PIL Image object or a Numpy array.
        """
        if isinstance(imdata, numpy.ndarray):
            imdata = Image.fromarray(numpy.uint8(imdata))
        elif isinstance(imdata, Image.Image):
            imdata.save(self._savepath(filename))

    def data(self, filename):
        image = Image.open(self._loadpath(filename))
        data = image.convert(self.data_mode)
        return {self.data_key:data}

    @classmethod
    def display(cls, value, size=256, format='png'):
        import Image
        # Return the base54 image if possible, else return None
        if isinstance(value, Image.Image):
            im = value
        elif cls.file_supported(value):
            im = Image.open(value)
        else:
            return None

        im.thumbnail((size,size))
        buff = StringIO()
        assert format=='png', "Only png display enabled"
        im.save(buff, format='png')
        buff.seek(0)
        return cls._img_tag(buff.read(), size=size)



class MatplotlibFile(FileType):
    """
    Since version 1.0, Matplotlib figures support pickling. An mpkl
    file is simply a pickled matplotlib figure.
    """

    extensions = param.List(default=['.mpkl'], constant=True)

    def __init__(self, **kwargs):
        super(MatplotlibFile, self).__init__(**kwargs)
        self.pprint_args(['hash_suffix'], [])

    def save(self, filename, fig):
        pickle.dump(fig, open(self._savepath(filename),'wb'))

    def metadata(self, filename):
        pklfile = open(self._loadpath(filename),'r')
        fig = pickle.load(pklfile)
        metadata = {'dpi':fig.dip, 'size':fig.size}
        pklfile.close()
        return metadata

    def data(self, filename):
        pklfile = open(self._loadpath(filename),'r')
        fig = pickle.load(pklfile)
        pklfile.close()
        return {self.data_key:fig}

    @classmethod
    def display(cls, value, size=256, format='png'):
        from matplotlib import pyplot
        if isinstance(value, pyplot.Figure):
            fig = value
        elif cls.file_supported(value):
            pklfile = open(value,'r')
            fig = pickle.load(pklfile)
        else:
            return None

        inches = size / float(fig.dpi)
        fig.set_size_inches(inches, inches)
        buff = StringIO()
        fig.savefig(buff, format=format)
        buff.seek(0)
        pyplot.close(fig)
        return cls._img_tag(buff.read(),
                            size=size,
                            format=format)



class SVGFile(FileType):
    """
    There is no standard way to handle SVG files in Python, therefore
    this class implements display only. For custom SVG handling, this
    can be subclassed by the user for a more complete implementation.
    """

    def save(self, filename, data):
        raise NotImplementedError

    def data(self, filename):
        raise NotImplementedError

    def metadata(self, filename):
        raise NotImplementedError

    @classmethod
    def display(cls,value, size=256, format='svg'):
        """
        SVG is a tricky format to support for saving and loading but
        it is easy to display.
        """
        if not isinstance(value, str): return None
        (_, ext) = os.path.splitext(value)
        if ext != '.svg':              return None
        data = open(value, 'r').read()
        return FileType._img_tag(data, size=size, format='svg')



class ViewFrame(param.ParameterizedFunction):
    """
    A FileViewer allows a DataFrame to be viewed as a HTML table
    containing image thumbnails in IPython Notebook.  Any filenames in
    the rows or columns of the DataFrame will be viewed as thumbnails
    according to the display functions in display_fns.

    Requires both Pandas and IPython Notebook.
    """

    size = param.Number(default=256, doc='''
       The size of the display image to generate.''')

    format = param.ObjectSelector(default='png', objects=['png','svg'],
       doc=''' The image format of the display. Either 'png' or 'svg'.''')

    display_fns = param.List(default=[ImageFile.display,
                                      MatplotlibFile.display,
                                      SVGFile.display],
       doc='''A list of display functions that return raster images
       (base64 encoded png images) based on filename or Python objects
       as input.''')


    def formatter(self,x):
        """
        A simple Pandas formatter that approximates the usual
        behaviour but also allows display of HTML images.
        """

        for dfn in self.display_fns:
            string = dfn(x, self.overrides.size, self.overrides.format)
            if string is not None:
                return string

        string = str(x)
        if len(string) <= self.max_colwidth:
            return string
        else:
            return string[:self.max_colwidth-3] + '...'


    def __call__(self, dframe, **kwargs):
        """
        Takes a DataFrame object and uses pandas to generate an HTML
        table containing display images.
        """
        import pandas
        from IPython.display import display, HTML

        self.overrides = param.ParamOverrides(self, kwargs)
        self.max_colwidth = pandas.get_option('max_colwidth')
        formatters = [self.formatter for el in range(len(dframe.columns))]
        # Pandas escapes contents by default (making this class necessary)
        pandas.set_option('max_colwidth',-1)
        html = dframe.to_html(escape=False, formatters=formatters)
        pandas.set_option('max_colwidth',self.max_colwidth)
        return display(HTML(html))

    def __repr__(self):
        return self._pprint(flat=True, annotate=False)

    def __str__(self):
        return self._pprint(annotate=False)
