/// License: GPLv3

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <chrono>

#include <unistd.h>
#include <signal.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <ctime>

#if CPU_BACKEND_ENABLED
# ifdef _OPENMP
#  define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_OMP
# else
#  define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CPP
# endif
#endif

#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/device_vector.h>

#if CPU_BACKEND_ENABLED
struct ushort4 {
	unsigned short x;
	unsigned short y;
	unsigned short z;
	unsigned short w;
};
void hipSetDevice(uint /* ignored */) {}
void hipMallocManaged(void**ptr, size_t sz)
{
	*ptr = malloc(sz);
}
#define hipFree free
#endif

#define FORCE_INLINE __attribute__((always_inline)) inline

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

static FORCE_INLINE __host__ __device__ __attribute__((pure)) const ushort& type(const particleinfo &info)
{ return info.x; }

static FORCE_INLINE __host__ __device__ __attribute__((pure)) uint id(const particleinfo &info)
{ return (uint)(info.z) | ((uint)(info.w) << 16); }

static FORCE_INLINE __host__ __device__ void set_id(particleinfo &info, const uint this_id)
{
	info.z = (this_id & 0xffff);
	info.w = (this_id >> 16);
}


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

struct FullViscSpec { };

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

const char *
thrust_device_system_name(int dev_system)
{
	switch (dev_system) {
#ifdef THRUST_DEVICE_SYSTEM_CUDA
	case THRUST_DEVICE_SYSTEM_CUDA: return "CUDA";
#endif
#ifdef THRUST_DEVICE_SYSTEM_OMP
	case THRUST_DEVICE_SYSTEM_OMP: return "OpenMP";
#endif
#ifdef THRUST_DEVICE_SYSTEM_TBB
	case THRUST_DEVICE_SYSTEM_TBB: return "Threading Building Blocks";
#endif
#ifdef THRUST_DEVICE_SYSTEM_CPP
	case THRUST_DEVICE_SYSTEM_CPP: return "C++";
#endif
#ifdef THRUST_DEVICE_SYSTEM_HIP
	case THRUST_DEVICE_SYSTEM_HIP: return "HIP";
#endif
	default: return "undefined";
	}
}

template<typename ViscSpec>
class Engine
{
public:

void
sort(particleinfo *info, hashKey *hash, uint *partidx, uint numParticles)
{
	static bool announced = false;
	if (!announced) {
		printf("Thrust v%d.%d.%dp%d, device system: %d (%s)\n",
			THRUST_MAJOR_VERSION, THRUST_MINOR_VERSION, THRUST_SUBMINOR_VERSION,
			THRUST_PATCH_NUMBER, THRUST_DEVICE_SYSTEM,
			thrust_device_system_name(THRUST_DEVICE_SYSTEM));
		announced = true;
	}

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
	CHECK("sort");
}

};

using namespace std;

void check(const char *file, unsigned long line, const char *func)
{
#if CPU_BACKEND_ENABLED
#else
	hipError_t err = hipDeviceSynchronize();
	if (hipSuccess != err) {
		stringstream errmsg;
		errmsg << file << ":" << line << " in " << func
			<< ": runtime API error " << err << " (" << hipGetErrorString(err) << ")";
		throw runtime_error(errmsg.str());
	}
#endif
}


int main(int argc, char *argv[])
{
	static constexpr uint numParticles = 84600;
	uint scale = 16;
	uint device = 0;

	const char * const * arg = argv + 1;
	while (argc > 1) {
		if (!strcmp(*arg, "--device")) {
			if (argc < 2)
				throw invalid_argument("please specify a device");
			--argc;
			++arg;
			device = atoi(*arg);
		} else if (!strcmp(*arg, "--scale")) {
			if (argc < 2)
				throw invalid_argument("please specify a scale");
			--argc;
			++arg;
			scale = atoi(*arg);
		}
		--argc;
		++arg;

	}
	const uint totalParticles = numParticles*scale;
	printf("Total particles: %u*%u=%u\n", numParticles, scale, totalParticles);

	hipSetDevice(device);
	CHECK("set device");

	particleinfo *info = NULL;
	hashKey *hash = NULL;
	uint *partidx = NULL;

	hipMallocManaged((void**)&info, totalParticles*sizeof(*info));
	hipMallocManaged((void**)&hash, totalParticles*sizeof(*hash));
	hipMallocManaged((void**)&partidx, totalParticles*sizeof(*partidx));

	cout << "Loading ..." << endl;

	ifstream data("data.idx");
	for (uint i = 0; i < numParticles; ++i) {
		particleinfo pi;
		data >> pi.x >> pi.y >> pi.z >> pi.w;

		info[i] = pi;

		data >> hash[i] >> partidx[i];
	}

	cout << "Init ..." << endl;

	for (uint loop = 1; loop < scale; ++loop) {
		const uint delta = loop*numParticles;
		for (uint i = 0; i < numParticles; ++i) {
			const uint dst_idx = i + delta;
			const particleinfo src = info[i];
			particleinfo &dst = info[dst_idx];

			dst.x = src.x;
			dst.y = src.y;
			set_id(dst, id(src) + delta);

			hash[dst_idx] = hash[i];
			partidx[dst_idx] = partidx[i] + delta;
		}
	}

	auto *engine = new Engine<FullViscSpec>();

	for (uint run = 0; run < 5; ++run) {
		cout << "Sorting ..." << endl;

		using clock = std::chrono::steady_clock;
		using time_point = clock::time_point;
		using duration = std::chrono::duration<double, std::milli>;

		time_point before = clock::now();
		engine->sort(info, hash, partidx, totalParticles);
		time_point after = clock::now();

		cout << "... done" << endl;
		cout << "Runtime: " << duration(after - before).count() << "ms" << endl;
	}

	hipFree(partidx);
	hipFree(hash);
	hipFree(info);
}
