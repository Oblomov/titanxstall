### Public domain

#override settings in Makefile.local
sinclude Makefile.local

# need for some substitutions
comma:=,
empty:=
space:=$(empty) $(empty)

CPPFLAGS +=-g -O3

CXXFLAGS += -std=c++14

CXXFLAGS += -Wall

ifeq ($(cpu),1)
 HIPCC = $(CXX)
 CPPFLAGS += -DCPU_BACKEND_ENABLED
else
 HIPCC = /opt/rocm/bin/hipcc
 HIPCCFLAGS += -x hip
 CPPFLAGS += -DHIP_BACKEND_ENABLED
endif
ifeq ($(openmp),1)
 CXXFLAGS += -fopenmp
 LDFLAGS += -fopenmp
endif

LINKER = $(HIPCC) $(HIPLDFLAGS)

PROG=thrust-test

all: $(PROG)

test: $(PROG)
	./$(PROG)
clean:
	rm -f $(PROG) $(PROG).o

%.o: %.hip.cc
	$(HIPCC) $(CXXFLAGS) $(CPPFLAGS) $(HIPCCFLAGS) -c -o $@ $<

%: %.o
	$(LINKER) -o $@ $^ $(LDFLAGS) $(LDLIBS)


.PRECIOUS: $(PROG).o $(PROG)
