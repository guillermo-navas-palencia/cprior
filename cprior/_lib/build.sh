#!/bin/sh

PROJECT="."
PROJECT_ROOT="$PROJECT"

SOURCE_DIR="$PROJECT_ROOT/src"
INCLUDE_DIR="$PROJECT_ROOT/include"

# compile
g++ -c -O3 -Wall -fpic -march=native -std=c++11 $SOURCE_DIR/*.cpp \
  -I $INCLUDE_DIR


# build library
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
  g++ -shared *.o -o _cprior.so

elif [[ "$OSTYPE" == "darwin"* ]]; then
  g++ -dynamiclib *.o -o cprior.dylib
fi

# move objects to build
mkdir -p build
mv *.o build/
