.. _installation:

Installation & Testing
======================

Install the current release of Netgraph with:

.. code-block:: shell

    pip install netgraph

To upgrade to a newer version, use the :code:`--upgrade` flag:

.. code-block::

    pip install --upgrade netgraph

If you do not have permission to install software systemwide, you can install into your user directory using the :code:`--user` flag:

.. code-block::

    pip install --user netgraph

If you are using (Ana-)conda (or mamba), you can also obtain netgraph from conda-forge:

.. code-block::

    conda install -c conda-forge netgraph

Alternatively, you can manually download netgraph from GitHub_ or PyPI_.
To install one of these versions, unpack it and run the following from the top-level source directory using the terminal:

.. _GitHub: https://github.com/paulbrodersen/netgraph
.. _PyPi: https://pypi.org/project/netgraph/

.. code-block::

    pip install .

Or without pip:

.. code-block::

    python setup.py install

Some of the examples in the documentation use additional libraries. These can be installed with:

.. code-block::

    pip install netgraph[docs]

For automated testing, install the additional dependencies required for testing using:

.. code-block::

    pip install netgraph[tests]

The full test suite can then be executed by running the following from the top-level source directory using the terminal:

.. code-block::

    pytest
