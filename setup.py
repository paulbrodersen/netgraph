from setuptools import setup, find_packages
setup(
    name = 'netgraph',
    version = '3.0.0',
    description = 'Fork of networkx drawing utilities for publication quality plots of networks.',
    author = 'Paul Brodersen',
    author_email = 'paulbrodersen+netgraph@gmail.com',
    url = 'https://github.com/paulbrodersen/netgraph',
    download_url = 'https://github.com/paulbrodersen/netgraph/archive/3.0.0.tar.gz',
    keywords = ['matplotlib', 'network', 'visualisation'],
    classifiers = [ # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Visualization'
    ],
    platforms=['Platform Independent'],
    packages=find_packages(),
    install_requires=['numpy', 'matplotlib'],
)
