#!/bin/bash

# Update Pybind11 submodule
git submodule init
git submodule update

if [ ! -d "build" ]; then
    mkdir build
fi

cd build
cmake -DCMAKE_C_COMPILER=/usr/bin/gcc-11 -DCMAKE_CXX_COMPILER=/usr/bin/g++-11 ..
make -j4
export PYTHONPATH="$PWD:$PYTHONPATH"
cd ..
