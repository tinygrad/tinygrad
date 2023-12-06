#!/bin/bash -e
TINYGRAD_WORKSPACE=$PWD
mkdir $TINYGRAD_WORKSPACE/test_external_dir
cd $TINYGRAD_WORKSPACE/test_external_dir
python3 -m venv venv
source venv/bin/activate
pip3 install $TINYGRAD_WORKSPACE
python3 -c "from tinygrad.tensor import Tensor; print(Tensor([1,2,3,4,5]))"
cd $TINYGRAD_WORKSPACE
rm -R $TINYGRAD_WORKSPACE/test_external_dir