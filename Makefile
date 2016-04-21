### Public domain

#override settings in Makefile.local
sinclude Makefile.local

# need for some substitutions
comma:=,
empty:=
space:=$(empty) $(empty)

#CUFLAGS ?=-arch=sm_52

ifneq ($(CXX),)
	CUFLAGS += -ccbin=$(CXX)
endif

CPPFLAGS +=-g

CXXFLAGS += -std=c++98 -O3

CXXFLAGS += -Wall

CUFLAGS += --compiler-options $(subst $(space),$(comma),$(strip $(CXXFLAGS))) -Xptxas="-v"

all: titanxstall titanxstall_moderngpu

titanxstall_moderngpu: titanxstall_moderngpu.cu
	nvcc -std=c++11 -Xptxas="-v" --expt-extended-lambda -arch sm_52 -I moderngpu/src -O2 -use_fast_math -o $@ $<


test: titanxstall
	./titanxstall
clean:
	rm -f titanxstall
	rm -f titanxstall_moderngpu

%: %.cu cached_alloc.h
	nvcc -arch sm_52 $(CPPFLAGS) $(CUFLAGS) $(LDFLAGS) $(LDLIBS) -o $@ $<

