# Titan X stall

This is a testing program to verify GPU stalls when running complex
thrust sorts on an NVIDIA Titan X GPU. I've first come across this issue
while working on the [GPUSPH][gpusph] codebase, hence the weird
structure of the test.

[gpusph]: http://www.gpusph.org

# Instructions

Build with `make`, run with `make test`. Only tested on Linux.

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
