/// Copyright (C) 2016 Giuseppe Bilotta <giuseppe.bilotta@gmail.com>
/// License: GPLv3

#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>

#define restrict __restrict__

typedef unsigned int uint;
typedef unsigned int hashKey;
typedef ushort4 particleinfo;

static __forceinline__ __host__ __device__ __attribute__((pure)) const ushort& type(const particleinfo &info)
{ return info.x; }

static __forceinline__ __host__ __device__ __attribute__((pure)) uint id(const particleinfo &info)
{ return (uint)(info.z) | ((uint)(info.w) << 16); }

#define PART_FLAG_SHIFT	3
#define PART_TYPE_MASK	((1<<PART_FLAG_SHIFT)-1)
#define PART_TYPE(f) (type(f) & PART_TYPE_MASK)

// some defines to make the compare functor and thrust sort key invokation more legible
typedef thrust::tuple<hashKey, particleinfo> hash_info_pair;

typedef thrust::device_ptr<particleinfo> thrust_info_ptr;
typedef thrust::device_ptr<hashKey> thrust_hash_ptr;
typedef thrust::device_ptr<uint> thrust_uint_ptr;

typedef thrust::tuple<thrust_hash_ptr, thrust_info_ptr> hash_info_iterator_pair;
typedef thrust::zip_iterator<hash_info_iterator_pair> key_iterator;

/// Functor to sort particles by hash (cell), and
/// by fluid number within the cell
struct ptype_hash_compare :
	public thrust::binary_function<hash_info_pair, hash_info_pair, bool>
{
	typedef thrust::tuple<hashKey, particleinfo> value_type;

	__host__ __device__
	bool operator()(const value_type& a, const value_type& b)
	{
		uint	ha(thrust::get<0>(a)),
			hb(thrust::get<0>(b));
		particleinfo	pa(thrust::get<1>(a)),
				pb(thrust::get<1>(b));

		if (ha == hb) {
			if (PART_TYPE(pa) == PART_TYPE(pb))
				return id(pa) < id(pb);
			return (PART_TYPE(pa) < PART_TYPE(pb));
		}
		return (ha < hb);
	}
};


void
sort(particleinfo *info, hashKey *hash, uint *partidx, uint numParticles)
{
	thrust_info_ptr particleInfo =
		thrust::device_pointer_cast(info);
	thrust_hash_ptr particleHash =
		thrust::device_pointer_cast(hash);
	thrust_uint_ptr particleIndex =
		thrust::device_pointer_cast(partidx);

	key_iterator key_start(thrust::make_tuple(particleHash, particleInfo));
	key_iterator key_end(thrust::make_tuple(
			particleHash + numParticles,
			particleInfo + numParticles));

        thrust::sort_by_key(key_start, key_end, particleIndex, ptype_hash_compare());
}

__global__ void
initParticles(
	particleinfo * restrict infoArray,
	hashKey * restrict hashArray,
	uint * restrict idxArray,
	uint numParticles)
{
	uint idx = threadIdx.x + blockIdx.x*blockDim.x;

	if (idx > numParticles)
		return;

	idxArray[idx] = idx;

	particleinfo info;
	info.x = idx % 4;
	info.y = 0;
	info.z = (ushort)(idx & 0xffff);
	info.w = (ushort)(idx >> 16);

	infoArray[idx] = info;

	hashArray[idx] = idx/17 + (idx % (idx & 17));
}

__global__ void
reHashParticles(
	particleinfo const * restrict infoArray,
	hashKey * restrict hashArray,
	uint const * restrict idxArray,
	uint numParticles)
{
	uint idx = threadIdx.x + blockIdx.x*blockDim.x;

	if (idx > numParticles)
		return;

	hashKey oldHash = hashArray[idx];
	const particleinfo info = infoArray[idx];
	const uint pid = id(info);

	hashArray[idx] = pid/17 + (oldHash % (pid & 17));
}

int main()
{
  uint numParticles = 5*1024*1024;
  unsigned long counter = 0;
  
  particleinfo *info = NULL;
  hashKey *hash = NULL;
  uint *partidx = NULL;
  
  cudaMalloc(&info, numParticles*sizeof(*info));
  cudaMalloc(&hash, numParticles*sizeof(*hash));
  cudaMalloc(&partidx, numParticles*sizeof(*partidx));
  
  const int blockSize = 1024;
  const int numBlocks = (numParticles + blockSize - 1)/blockSize;
  
  while(true)
  {
    initParticles<<<numBlocks, blockSize>>>(info, hash, partidx, numParticles);
    
    sort(info, hash, partidx, numParticles);
    
    ++counter;
    
    if(counter % 1000 == 0) std::cout << "Iteration " << counter << std::endl;
  }
}

