Getting started
===============

Installation
------------

Install release
"""""""""""""""

You can download the latest release for your OS and install using

.. code-block:: bash

   python setup.py install

Install from source
"""""""""""""""""""

To install from source, download the git repository https://github.com/guillermo-navas-palencia/cprior. Then, you need to build a shared library (_cprior.so) for Linux or a dynamic-link library (cprior.dll) for Windows, before running ``python setup.py install``. For example. for Linux you can use the bash script ``cprior/_lib/build.sh``

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

For Windows, you can use Visual Studio Express or CodeBlocks to build your DLL.