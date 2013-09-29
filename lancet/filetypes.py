import os, tempfile, StringIO, json
import param
#from core import Args
import numpy as np
from core import PrettyPrinted


class buff_img(object):
    """
    A small context manager that helps build an HTML display image
    via any utility that can save an image as a png file.
    """
    def __enter__(self):
        self.buff = StringIO.StringIO()
        self.html = ''
        return self.buff 
    
    def __exit__(self, *args):
        self.buff.seek(0)
        prefix = 'data:image/png;base64,'
        b64 = prefix+self.buff.read().encode("base64")
        self.html = '<img src="%s" />' % b64

class mpl_img(object):
    """
    A small context manager that helps build an HTML display image
    using matplotlb. Simply call pyplot plotting commands within the
    block and the context manager will handle closing the figure
    (which is always accessible via the fig attribute).

    Useful for quickly building custom display functions for arbitrary
    file data using Matplotlib.
    """

    def __init__(self, size):
        self.size = size

    def __enter__(self):
        import matplotlib.pyplot as plt
        global plt
        self.buff = StringIO.StringIO()
        self.fig = plt.figure()
        inches = self.size / float(self.fig.dpi)
        self.fig.set_size_inches(inches, inches)
        self.html = ''
        return self.buff 
    
    def __exit__(self, *args):
        self.fig.savefig(self.buff, format='png')
        self.buff.seek(0)
        prefix = 'data:image/png;base64,'
        b64 = prefix+self.buff.read().encode("base64")
        plt.close(self.fig)
        self.html = '<img src="%s" />' % b64



class FileType(PrettyPrinted, param.Parameterized):
    """
    The base class for all supported file types in Lancet. This class
    is designed to be simple and easily extensible to support new
    files and has only three essential methods: 'save', 'data' and
    'metadata').

    Optionally, a 'display' classmethod may be implemented to
    allow the file contents to be quickly visualized.
    """

    hash_suffix = param.Boolean(default=True, doc="""
       Whether to ensure the saved filename is unique by adding a
       short hash suffix. Note that this is a class level parameter
       only.""")

    directory = param.String(default='.', allow_None=True, doc="""
       Directory in which to load or save the file. Note that this
       is a class level parameter only.""")

    extensions = param.List(default=[], constant=True, 
       doc= """The set of supported file extensions.""")

    data_key = param.String(default='data', doc="""
       The name (key) given to the file contents if the key cannot be
       determined from the file itself.""")

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
        realpath = os.path.realpath(os.path.join(self.directory, filename))
        return filename if os.path.isfile(filename) else realpath

    def _savepath(self, filename):
        """
        Returns the full path for saving the file, adding an extension
        and making the filename unique as necessary.
        """
        (basename, ext) = os.path.splitext(filename)
        ext = ext if ext else self.extensions[0]
        savepath = os.path.realpath(os.path.join(self.directory, 
                                                 '%s%s' % (basename, ext)))
        return (tempfile.mkstemp(ext, basename + "_", self.directory)[1] 
                if self.hash_suffix else savepath)

    @classmethod
    def file_supported(cls, filename):
        if not isinstance(filename, str):
            return False
        (_, ext) = os.path.splitext(filename)
        if ext not in cls.extensions: 
            return False
        else:
            return True

    @classmethod
    def display(cls, cellval, size=64):
        """
        Returns a function that generates a display image as a base64
        encoded HTMl image (png format). Should be implemented if
        there is an obvious visualization of the file type.
        """
        if cls.file_supported(cellval):
            # Return the base54 image if possible, else return cellval
            return cellval
        else:
            return cellval

    def __repr__(self):
        return self._pprint(flat=True, annotate=False)

    def __str__(self):
        return self._pprint(annotate=False)

    def __or__(self, other):
        return FileOption(self, other)


class CustomFileLoader(FileType):
    
    data_fn = param.Callable()
    
    metadata_fn = param.Callable()
    
    def __init__(self, data_fn=None, metadata_fn=None, **kwargs):
        zipped = zip(['data_fn','metadata_fn'], [data_fn, metadata_fn])
        fn_dict = dict([(k,v) for (k,v) in zipped if (v is not None)])
        super(CustomFileLoader, self).__init__(**dict(kwargs, **fn_dict))
        
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
    

class FileOption(FileType):
    """
    Allows one FileType in a pair of FileTypes to handle a given
    file. For instance, given a mixed list of image files and numpy
    npz files, the ImageFile() | NumpyFile() object an handle either
    type of file appropriately.
    """
    first = param.ClassSelector(class_=FileType)

    second = param.ClassSelector(class_=FileType)

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
        return json.load(jsonfile)
    
    def data(self, filename):
        raise Exception("JSONFile only loads metadata")


class NumpyFile(FileType):

    extensions = param.List(default=['.npz'], constant=True)

    compress = param.Boolean(default=True)

    def __init__(self, **kwargs):
        import numpy as np
        global np
        super(NumpyFile, self).__init__(**kwargs)
        self.pprint_args(['hash_suffix'], ['compress']) # CHECK!

    def save(self, filename, metadata={}, **data):
        super(NumpyFile, self).save(filename, metadata, **data)
        savefn = np.savez_compressed if self.compress else np.savez
        savefn(self._savepath(filename), metadata=metadata, **data)

    def metadata(self, filename):
        npzfile = np.load(self._loadpath(filename))
        metadata = (npzfile['metadata'].tolist() 
                    if 'metadata' in npzfile.keys() else {})
        # Numpy load may return a Python dictionary.
        if not isinstance(npzfile, dict): npzfile.close()
        return metadata

    def data(self, filename):
        npzfile = np.load(self._loadpath(filename))
        keys = [k for k in npzfile.keys() if k != 'metadata']
        data = dict((k,npzfile[k]) for k in keys)
        
        # IS THIS SAFE?
        for (k,val) in data.items():
            if val.dtype.char == 'O' and val.shape == ():
                data[k] = val[()]

        if not isinstance(npzfile, dict):
            npzfile.close()
        return data



class ImageFile(FileType):
    """
    Requires PIL or Pillow.
    """
    extensions = param.List(default=['.png', '.jpg'], constant=True)

    image_info = param.Dict(default={'mode':'mode', 'size':'size', 'format':'format'}, 
        doc="""Dictionary of the metadata to load. Each key is the
        name given to the metadata item and each value is the PIL
        Image attribute to return.""")

    data_mode = param.ObjectSelector(default='RGBA', 
                                     objects=['L', 'RGB', 'RGBA', 'I','F'], 
        doc="""Sets the mode of the mode of the Image object. Palette
        mode'P is not supported""")

    data_key = param.String(default='images', doc="""
       The name (key) given to the loaded image data.""")

    def __init__(self, **kwargs):
        import Image
        global Image
        super(ImageFile, self).__init__(**kwargs)
        self.pprint_args(['hash_suffix'], ['data_key', 'data_mode', 'image_info'])

    def metadata(self, filename, **kwargs):
        image = Image.open(self._loadpath(filename))
        return dict((name, getattr(image,attr,None))
                    for (name, attr) in self.image_info.items())

    def save(self, filename, imdata, **data):
        """
        Data may be either a PIL Image object or a Numpy array.
        """
        if isinstance(imdata, np.ndarray):
            imdata = Image.fromarray(np.uint8(imdata))
        elif isinstance(imdata, Image.Image):
            imdata.save(self._savepath(filename))

    def data(self, filename):
        image = Image.open(self._loadpath(filename))
        data = image.convert(self.data_mode)
        return {self.data_key:data}

    @classmethod
    def display(cls, cellval, size=256):
        import Image
        if cls.file_supported(cellval):
            im = Image.open(cellval)
            im.thumbnail((size,size))
            img = buff_img()
            with img as f:  im.save(f, format='png')
            return img.html
        else:
            return cellval


class ViewFrame(param.ParameterizedFunction):
    """
    A FileViewer allows a DataFrame (or corresponding Args object) to
    be viewed as a HTML table containing image thumbnails in IPython
    Notebook.  Any filenames in the rows or columns of the DataFrame
    will be viewed thumbnails according to the display functions in
    display_by_filename. Object type can also be displayed according
    to the display_by_type dictionary.

    Requires both Pandas and IPython Notebook.
    """

    size = param.Number(default=256, doc="""
       The size of the display image to generate.""")

    display_by_filename = param.List(default=[ImageFile.display])

    display_by_type = param.Dict(default={})

    def formatter(self,x):
        """
        A simple Pandas formatter that approximates the usual
        behaviour but also allows display of HTML images.
        """
        
        if isinstance(x, str):
            for dfn in self.display_by_filename:
                string = dfn(x, self.size)
                if string.startswith("<img src="):
                    return string
        elif type(x) in self.display_by_type:
            dfn = self.display_by_type[type(x)]
            return dfn(x, self.size)
        
        string = str(x)
        if len(string) <= self.max_colwidth:
            return string
        else:
            return string[:self.max_colwidth-3] + '...'


    def __call__(self, dframe, **kwargs): # size=None
        """
        Takes a DataFrame object and uses pandas to generate an HTML
        table containing display images.
        """
        import pandas
        from IPython.display import display, HTML

        p= param.ParamOverrides(self, kwargs)
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
