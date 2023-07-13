# ROCm Thrust performance

This is a testing program to check for the custom sort performance
used in he [GPUSPH][gpusph] codebase,
hence the weird structure of the test.

[gpusph]: http://www.gpusph.org

# Instructions

Build with `make`, run with `make test`. Only tested on Linux.

By default this builds using HIP for AMD GPU, you can build for CPU with
`make cpu=1`, and with OpenMP (recommended on CPU) with `make openmp=1`.

Typical comparison can be done running:

```
make clean ; make test
make clean ; make cpu=1 openmp=1 test
```
