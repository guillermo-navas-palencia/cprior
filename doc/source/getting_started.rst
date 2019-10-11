Getting started
===============

Installation
------------

Install release
"""""""""""""""

To install the current release of CPrior on Linux/Windows:

.. code-block:: bash

   pip install cprior

Optionally, download a different release
from https://github.com/guillermo-navas-palencia/cprior/releases and install
using

.. code-block:: bash

   python setup.py install

Install from source
"""""""""""""""""""

To install from source, download the git repository https://github.com/guillermo-navas-palencia/cprior. Then, build a shared library (_cprior.so) for Linux or a dynamic-link library (cprior.dll) for Windows, before running ``python setup.py install``. For example. for Linux you might use the bash script ``cprior/_lib/build.sh``

.. code-block:: bash

   #!/bin/sh

   PROJECT="."
   PROJECT_ROOT="$PROJECT"

   SOURCE_DIR="$PROJECT_ROOT/src"
   INCLUDE_DIR="$PROJECT_ROOT/include"

   # compile
   g++ -c -O3 -Wall -fpic -march=native -std=c++11 $SOURCE_DIR/*.cpp -I $INCLUDE_DIR

   # build library
   g++ -shared *.o -o _cprior.so

For Windows, the easiest is to use the above bash script with Cygwin
replacing ``g++`` by ``x86_64-w64-mingw32-g++``. Alternatively, you might use
Visual Studio Express or CodeBlocks to build your DLL. Once built
just copy it to ``cprior/_lib/``.


Dependencies
------------
CPrior has been tested with CPython 3.5, 3.6 and 3.7. It requires:

* mpmath 1.0.0 or later. Website: http://mpmath.org/
* numpy 1.15.0 or later. Website: https://www.numpy.org/
* scipy 1.0.0 or later. Website: https://scipy.org/scipylib/
* pytest
* coverage

Note that older versions might work correctly. Run tests to verify that all unit
tests pass.


Testing
-------

To run all unit tests use:

.. code-block:: bash

   python setup.py test


Example
-------

A Bayesian A/B test with data following a Bernoulli distribution with two
distinct success probability. This example is a simple use case for
CRO (conversion rate) or CTR (click-through rate) testing.

   >>> from scipy import stats
   >>> from cprior import BernoulliABTest, BernoulliModel
   >>> modelA = BernoulliModel()
   >>> modelB = BernoulliModel()
   >>> abtest = BernoulliABTest(modelA=modelA, modelB=modelB, simulations=1000000)

Generate new data and update models

   >>> data_A = stats.bernoulli(p=0.10).rvs(size=1500, random_state=42)
   >>> data_B = stats.bernoulli(p=0.11).rvs(size=1600, random_state=42)
   >>> abtest.update_A(data_A)
   >>> abtest.update_B(data_B)

Compute :math:`P[A > B]`

   >>> abtest.probability(variant="A")
   0.10243749066178826

Compute :math:`P[B > A]`:

   >>> abtest.probability(variant="B")
   0.8975625093382118

Compute posterior expected loss :math:`\mathrm{E}[\max(B - A, 0)]`

   >>> abtest.expected_loss(variant="A")
   0.014747280681722819

and :math:`\mathrm{E}[\max(A - B, 0)]`

   >>> abtest.expected_loss(variant="B")
   0.0005481520957841303
