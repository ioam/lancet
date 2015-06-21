#!/usr/bin/env python

import sys, os
from distutils.core import setup

setup_args = {}
install_requires = ['param>=0.0.1']


setup_args.update(dict(
    name='lancet-ioam',
    version="0.9.0",
    install_requires = install_requires,
    description='Launch jobs, organize the output, and dissect the results.',
    long_description=open('README.rst').read() if os.path.isfile('README.rst') else 'Consult README.rst',
    author= "Jean-Luc Stevens and Marco Elver",
    author_email= "developers@topographica.org",
    maintainer= "IOAM",
    maintainer_email= "developers@topographica.org",
    platforms=['Windows', 'Mac OS X', 'Linux'],
    license='BSD',
    url='http://ioam.github.com/lancet/',
    packages = ["lancet"],
    classifiers = [
        "License :: OSI Approved :: BSD License",
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries"]
))


def check_pseudo_package(path):
    """
    Verifies that a fake subpackage path for assets (notebooks, svgs,
    pngs etc) both exists and is populated with files.
    """
    if not os.path.isdir(path):
        raise Exception("Please make sure pseudo-package %s exists." % path)
    else:
        assets = os.listdir(path)
        if len(assets) == 0:
            raise Exception("Please make sure pseudo-package %s is populated." % path)

if __name__=="__main__":

    if 'LANCET_RELEASE' in os.environ:
        # Add unit tests
        setup_args['packages'].append('lancet.tests')

        if ('upload' in sys.argv) or ('sdist' in sys.argv):
            check_pseudo_package(os.path.join('.', 'lancet', 'tests'))
            import lancet
            lancet.__version__.verify(setup_args['version'])

    setup(**setup_args)
