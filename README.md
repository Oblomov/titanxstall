# Titan X stall

This is a testing program to verify GPU stalls when running complex
thrust sorts on an NVIDIA Titan X GPU. I've first come across this issue
while working on the [GPUSPH][gpusph] codebase, hence the weird
structure of the test.

[gpusph]: http://www.gpusph.org

# Instructions

Build with `make`, run with `make test`. Only tested on Linux.

You can customize build options by adding a `Makefile.local` file. This can be
used e.g. to override the compute capability for which the program gets build
(e.g. set `CUFLAGS=-arch=sm_20` to build for a Fermi card).

If you want to run the program on a different device than the default and/or
with a different number of elements than the default (5Mi), launch it manually.
For example, to run on the 4th device with 10Mi elements, use:

   ./titanxstall --device 3 --elements $((10*1024*1024))

(for example).

The `--cache-alloc` option can be passed to `titanxstall` to use a custom
caching allocator in thrust in place of the default policy.

# Results

The program runs forever, or until it crashes or the GPU stalls.

At every iteration, it will change some of the hashes and sort again.

During execution, a file in `/dev/shm` will keep you up to date on the
current progress. Every thousandth iteration, something will also be
shown on console.

If the info stream in `/dev/shm` doesn't update anymore, the GPU is
stalling. If it's attached to a display (or has a watchdog enabled for
any other reason), the GPU reset will trigger an error and crash the
program, but if no watchdog is active, the program will just get stuck
at the same iteration, so the only way to detect this has happened is by
looking at the info stream.
