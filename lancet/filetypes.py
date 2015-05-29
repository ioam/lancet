import os, tempfile, json, pickle
import param
from lancet.core import PrettyPrinted

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

    def __init__(self, **params):
        super(FileType, self).__init__(**params)
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
        basename = basename if (ext in self.extensions) else filename
        ext = ext if (ext in self.extensions) else self.extensions[0]
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

    def __init__(self, first, second, **params):
        if set(first.extensions) & set(second.extensions):
            raise Exception("FileTypes must support non-overlapping sets of extensions.")
        extensions = set(first.extensions) | set(second.extensions)
        super(FileOption, self).__init__(first=first, second=second,
                                         extensions = list(extensions), **params)
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

    def __init__(self, data_fn=None, metadata_fn=None, **params):
        zipped = zip(['data_fn','metadata_fn'], [data_fn, metadata_fn])
        fn_dict = dict([(k,v) for (k,v) in zipped if (v is not None)])
        super(CustomFile, self).__init__(**dict(params, **fn_dict))

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


class HVZFile(CustomFile):
    """
    FileType supporting the .hvz file format of the the HoloViews
    library (http://ioam.github.io/holoviews).

    Equivalent to the following CustomFile:

    CustomFile(metadata_fn=lambda f: Unpickler.key(f),
               data_fn = lambda f: {e: Unpickler.load(f, [e])
                                       for e in Unpickler.entries(f)})
    """

    def hvz_data_fn(f):
        from holoviews.core.io import Unpickler
        return {e: Unpickler.load(f, [e]) for e in Unpickler.entries(f)}

    def hvz_metadata_fn(f):
        from holoviews.core.io import Unpickler
        return Unpickler.key(f)

    data_fn = param.Callable(hvz_data_fn, doc="""
        By default loads all the entries in the .hvz file using
        Unpickler.load and returns them as a dictionary.""")

    metadata_fn = param.Callable(hvz_metadata_fn, doc="""
       Returns the key stored in the .hvz file as metadata using the
       Unpickler.key method.""")



class JSONFile(FileType):
    """
    It is assumed you won't store very large volumes of data as JSON.
    For this reason, the contents of JSON files are loaded as
    metadata.
    """
    extensions = param.List(default=['.json'], constant=True)

    def __init__(self, **params):
        super(JSONFile, self).__init__(**params)
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

    def __init__(self, **params):
        super(NumpyFile, self).__init__(**params)
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

    def __init__(self, **params):
        super(ImageFile, self).__init__(**params)
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


class MatplotlibFile(FileType):
    """
    Since version 1.0, Matplotlib figures support pickling. An mpkl
    file is simply a pickled matplotlib figure.
    """

    extensions = param.List(default=['.mpkl'], constant=True)

    def __init__(self, **params):
        super(MatplotlibFile, self).__init__(**params)
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
