# l0_physops
This repo represents a shared library to perform operations of a physical query execution plan (e.g., build a hash table for a join) on heterogeneous hardware (here Intel's L0 is mainly targeted).
This shared library can be linked to an existing projects (checked on [HDK](https://github.com/intel-ai/hdk)).

# Requirements
```
conda create -n dpcpp-dev
conda activate dpcpp-dev
conda install -c conda-forge dpcpp_impl_linux-64
```
# Setup & Building
Once in `l0_physops` directory:

Setup:
```
mkdir build
cd build
cmake ..
```
Building:
```
make
```
What you get:

For example: `l0_physops/build/hash_table/libhash_table.so`.
This `.so` can be linked into a project.
