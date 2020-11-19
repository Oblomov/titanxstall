### Public domain

#override settings in Makefile.local
sinclude Makefile.local

# need for some substitutions
comma:=,
empty:=
space:=$(empty) $(empty)

CUFLAGS ?=-arch=sm_75

ifneq ($(CXX),)
	CUFLAGS += -ccbin=$(CXX)
endif

NVCC ?= nvcc

CPPFLAGS +=-g -O3

CXXFLAGS += -std=c++14

CXXFLAGS += -Wall

CUFLAGS += --compiler-options $(subst $(space),$(comma),$(strip $(CXXFLAGS)))
CUFLAGS += -lineinfo

LINKER = $(NVCC) $(CUFLAGS)

PROG=thrust-cuda11-sort-bug
all: $(PROG)

test: $(PROG)
	./$(PROG)
clean:
	rm -f $(PROG) $(PROG).o

%.o: %.cu
	$(NVCC) $(CPPFLAGS) $(CUFLAGS) $(LDFLAGS) $(LDLIBS) -c -o $@ $<

%: %.o
	$(LINKER) -o $@ $^ $(LDFLAGS) $(LDLIBS)


.PRECIOUS: $(PROG).o $(PROG)
