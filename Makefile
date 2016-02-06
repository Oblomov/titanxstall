### Public domain

#override settings in Makefile.local
sinclude Makefile.local

# need for some substitutions
comma:=,
empty:=
space:=$(empty) $(empty)

CUFLAGS ?=-arch=sm_52

ifneq ($(CXX),)
	CUFLAGS += -ccbin=$(CXX)
endif

CPPFLAGS +=-g -O3

CXXFLAGS += -std=c++98

CXXFLAGS += -Wall

CUFLAGS += --compiler-options $(subst $(space),$(comma),$(strip $(CXXFLAGS)))

all: titanxstall

test: titanxstall
	./titanxstall
clean:
	rm -f titanxstall

%: %.cu cached_alloc.h
	nvcc $(CPPFLAGS) $(CUFLAGS) $(LDFLAGS) $(LDLIBS) -o $@ $<

