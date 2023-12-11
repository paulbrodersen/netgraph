import os.path

from setuptools import setup, find_packages

def read_file(filename):
    with open(os.path.join(os.path.dirname(__file__), filename)) as file:
        return file.read()

version = '4.13.2'

setup(
    name='netgraph',
    version=version,
    description='Python drawing utilities for publication quality plots of networks.',
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
    author='Paul Brodersen',
    author_email='paulbrodersen+netgraph@gmail.com',
    url='https://github.com/paulbrodersen/netgraph',
    download_url=f'https://github.com/paulbrodersen/netgraph/archive/{version}.tar.gz',
    keywords=['matplotlib', 'network', 'visualisation'],
    classifiers=[ # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Visualization'
    ],
    platforms=['Platform Independent'],
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=['numpy', 'matplotlib', 'scipy', 'rectangle-packer', 'grandalf'],
    extras_require={
        'tests' : ['pytest', 'pytest-mpl'],
        'docs' : ['sphinx', 'sphinx-rtd-theme', 'numpydoc', 'sphinx-gallery', 'Pillow', 'networkx'],
    },
)
