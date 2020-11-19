/// Copyright (C) 2016 Giuseppe Bilotta <giuseppe.bilotta@gmail.com>
/// License: GPLv3

#include <sstream>
#include <fstream>
#include <stdexcept>

#include <unistd.h>
#include <signal.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <ctime>

#include <cuda_runtime.h>

#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/system/cuda/execution_policy.h>

// declare here for use by cached_alloc.h, should be refactored
void check(const char *file, unsigned long line, const char *func);
#define CHECK(func) check(__FILE__, __LINE__, func)

#define restrict __restrict__

typedef unsigned int uint;
typedef unsigned int hashKey;
typedef ushort4 particleinfo;

enum ParticleType {
	PT_FLUID = 0,
	PT_BOUNDARY,
	PT_VERTEX,
	PT_TESTPOINT,
	PT_NONE
};

static __forceinline__ __host__ __device__ __attribute__((pure)) const ushort& type(const particleinfo &info)
{ return info.x; }

static __forceinline__ __host__ __device__ __attribute__((pure)) uint id(const particleinfo &info)
{ return (uint)(info.z) | ((uint)(info.w) << 16); }

#define PART_FLAG_SHIFT	3
#define PART_TYPE_MASK	((1<<PART_FLAG_SHIFT)-1)
#define PART_TYPE(f) ParticleType(type(f) & PART_TYPE_MASK)

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
			const ParticleType pta = PART_TYPE(pa),
				ptb = PART_TYPE(pb);
			if (pta == ptb)
				return id(pa) < id(pb);
			return (pta < ptb);
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

	ptype_hash_compare comp;

	key_iterator key_start(thrust::make_tuple(particleHash, particleInfo));
	key_iterator key_end(thrust::make_tuple(
			particleHash + numParticles,
			particleInfo + numParticles));

	thrust::sort_by_key(key_start, key_end, particleIndex, comp);
}

using namespace std;

void check(const char *file, unsigned long line, const char *func)
{
	cudaError_t err = cudaDeviceSynchronize();
	if (cudaSuccess != err) {
		stringstream errmsg;
		errmsg << file << ":" << line << " in " << func
			<< ": runtime API error " << err << " (" << cudaGetErrorString(err) << ")";
		throw runtime_error(errmsg.str());
	}
}


int main(int argc, char *argv[])
{
	uint numParticles = 84600;
	uint device = 0;

	const char * const * arg = argv + 1;
	while (argc > 1) {
		if (!strcmp(*arg, "--device")) {
			if (argc < 2)
				throw invalid_argument("please specify a device");
			--argc;
			++arg;
			device = atoi(*arg);
		}
		--argc;
		++arg;

	}

	cudaSetDevice(device);
	CHECK("set device");

	particleinfo *info = NULL;
	hashKey *hash = NULL;
	uint *partidx = NULL;

	cudaMallocManaged(&info, numParticles*sizeof(*info));
	cudaMallocManaged(&hash, numParticles*sizeof(*hash));
	cudaMallocManaged(&partidx, numParticles*sizeof(*partidx));

	ifstream data("data.idx");
	for (uint i = 0; i < numParticles; ++i) {
		particleinfo pi;
		data >> pi.x >> pi.y >> pi.z >> pi.w;

		info[i] = pi;

		data >> hash[i] >> partidx[i];
	}

	for (uint i = 0; i < numParticles; ++i) {
		printf("BEFORE: %d: %d %d %d %d %d %d\n", i, info[i].x, info[i].y, info[i].z, info[i].w,
			hash[i], partidx[i]);
	}

	sort(info, hash, partidx, numParticles);
	CHECK("sort"); // sync

	for (uint i = 0; i < numParticles; ++i) {
		printf("AFTER: %d: %d %d %d %d %d %d\n", i, info[i].x, info[i].y, info[i].z, info[i].w,
			hash[i], partidx[i]);
	}

}

