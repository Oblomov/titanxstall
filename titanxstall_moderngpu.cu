/// Copyright (C) 2016 Giuseppe Bilotta <giuseppe.bilotta@gmail.com>
/// License: GPLv3

#include <iostream>
#include <moderngpu/kernel_mergesort.hxx>
#include <moderngpu/transform.hxx>
#include <moderngpu/tuple.hxx>

#define restrict __restrict__

typedef unsigned int uint;
typedef unsigned int hashKey;
typedef ushort4 particleinfo;

MGPU_HOST_DEVICE ushort type(particleinfo info)
{ return info.x; }

MGPU_HOST_DEVICE uint id(particleinfo info)
{ return (uint)(info.z) | ((uint)(info.w) << 16); }

#define PART_FLAG_SHIFT	3
#define PART_TYPE_MASK	((1<<PART_FLAG_SHIFT)-1)
#define PART_TYPE(f) (type(f) & PART_TYPE_MASK)

using namespace mgpu;

int main()
{
  standard_context_t context;

  uint numParticles = 5*1024*1024;
  
  typedef tuple<hashKey, particleinfo> key_t;
  mem_t<key_t> keys(numParticles, context);
  mem_t<int> partidx(numParticles, context);

  int counter = 0;
    
  while(true)
  {
    // Prepare the things to sort.
    key_t* keys_data = keys.data();
    int* partidx_data = partidx.data();

    transform([=]MGPU_DEVICE(int index) {
      partidx_data[index] = index;

      // Include some hash of the counter.
        keys_data[index] = key_t(
          index / 17 + (counter % (17 & index)),
        particleinfo {
          (ushort)(index % 4),
          0,
          (ushort)(0xffff & index),
          (ushort)(index>> 16)
        }
      );
    }, numParticles, context);
  
    // returns a < b.
    auto comp = []MGPU_DEVICE(key_t a, key_t b) -> bool {
      auto ha = get<0>(a), hb = get<0>(b);
      auto pa = get<1>(a), pb = get<1>(b);
      bool result;
      if(ha == hb) {
        if(PART_TYPE(pa) == PART_TYPE(pb))
          result = id(pa) < id(pb);
        else
          result = PART_TYPE(pa) < PART_TYPE(pb);
      } else
        result = ha < hb;
      return result;
    };

    // This is a big structure so choose a smaller grain size than the default.
    typedef launch_params_t<256, 3> launch_t;
    mergesort<launch_t>(keys_data, partidx_data, numParticles, comp, context);
    
    if(0 == counter % 100) printf("iteration %d\n", counter);
    ++counter;
  }
}


