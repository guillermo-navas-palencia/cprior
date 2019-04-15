#!/bin/sh

PROJECT="."
PROJECT_ROOT="$PROJECT"

SOURCE_DIR="$PROJECT_ROOT/src"
INCLUDE_DIR="$PROJECT_ROOT/include"

# compile
g++ -c -O3 -Wall -fpic -march=native -std=c++11 $SOURCE_DIR/*.cpp \
  -I $INCLUDE_DIR

# build library
g++ -shared *.o -o _cprior.so

# move objects to build
mkdir -p build
mv *.o build/
