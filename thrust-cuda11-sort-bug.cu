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

enum SPHFormulation { SPH_F1 = 1, };

enum BoundaryType { LJ_BOUNDARY, };

enum Periodicity { PERIODIC_NONE = 0, };

typedef uint64_t flag_t;

#define ENABLE_NONE ((flag_t)0)

enum RheologyType { NEWTONIAN, };

enum TurbulenceModel { LAMINAR_FLOW, KEPSILON, };

enum ComputationalViscosityType { KINEMATIC, };

enum ViscousModel { MORRIS, };

enum AverageOperator { ARITHMETIC, };

template<
	RheologyType _rheologytype = NEWTONIAN,
	TurbulenceModel _turbmodel = LAMINAR_FLOW,
	ComputationalViscosityType _compvisc = KINEMATIC,
	ViscousModel _viscmodel = MORRIS,
	AverageOperator _avgop = ARITHMETIC,
	flag_t _simflags = ENABLE_NONE,
	// is this a constant-viscosity formulation?
	bool _is_const_visc = (
		(_simflags != ENABLE_NONE) &&
		(_rheologytype == NEWTONIAN) &&
		(_turbmodel != KEPSILON)
	)
>
struct FullViscSpec {
	static constexpr RheologyType rheologytype = _rheologytype;
	static constexpr TurbulenceModel turbmodel = _turbmodel;
	static constexpr ComputationalViscosityType compvisc = _compvisc;
	static constexpr ViscousModel viscmodel = _viscmodel;
	static constexpr AverageOperator avgop = _avgop;
	static constexpr flag_t simflags = _simflags;

	static constexpr bool is_const_visc = _is_const_visc;
};

class AbstractEngine
{
public:
	virtual void sort(particleinfo *info, hashKey *hash, uint *partidx, uint numParticles) = 0;
};

template<SPHFormulation sph_formulation, typename ViscSpec, BoundaryType boundarytype, Periodicity periodicbound, flag_t simflags,
	bool neibcount>
class CUDAEngine : public AbstractEngine
{
public:

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

	for (uint i = 0; i < numParticles; ++i) {
		printf("BEFORE: %d: %d %d %d %d %d %d\n", i, info[i].x, info[i].y, info[i].z, info[i].w,
			hash[i], partidx[i]);
	}

	ptype_hash_compare comp;

	key_iterator key_start(thrust::make_tuple(particleHash, particleInfo));
	key_iterator key_end(thrust::make_tuple(
			particleHash + numParticles,
			particleInfo + numParticles));

	if (numParticles > 0)
		thrust::sort_by_key(key_start, key_end, particleIndex, comp);

	for (uint i = 0; i < numParticles; ++i) {
		printf("AFTER: %d: %d %d %d %d %d %d\n", i, info[i].x, info[i].y, info[i].z, info[i].w,
			hash[i], partidx[i]);
	}
}

};

__global__ void
initIdx(uint* partidx, uint numParticles)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles)
		return;

	partidx[index] = index;
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

	particleinfo *h_info = NULL, *d_info = NULL;
	hashKey *h_hash = NULL, *d_hash = NULL;
	uint *h_partidx = NULL, *d_partidx = NULL;

	float4 *d_pos = NULL, *d_vel = NULL, *d_forces = NULL;

	uint *d_cellStart = NULL, *d_cellEnd = NULL;
	ushort *d_neibsList = NULL;

	uint numCells = 10455;
	uint neibsListSize = numParticles*128;


	h_info = new particleinfo[numParticles];
	h_hash = new hashKey[numParticles];
	h_partidx = new uint[numParticles];

	cudaMallocManaged(&d_pos, numParticles*sizeof(*d_pos));
	cudaMallocManaged(&d_vel, numParticles*sizeof(*d_vel));
	cudaMallocManaged(&d_info, numParticles*sizeof(*d_info));

	cudaMallocManaged(&d_forces, numParticles*sizeof(*d_forces));
	cudaMemset(d_forces, 0, numParticles*sizeof(*d_forces));

	cudaMallocManaged(&d_cellStart, numCells*sizeof(*d_cellStart));
	cudaMemset(d_cellStart, -1, numCells*sizeof(*d_cellStart));
	cudaMallocManaged(&d_cellEnd, numCells*sizeof(*d_cellEnd));
	cudaMemset(d_cellEnd, -1, numCells*sizeof(*d_cellEnd));

	cudaMallocManaged(&d_hash, numParticles*sizeof(*d_hash));
	cudaMallocManaged(&d_partidx, numParticles*sizeof(*d_partidx));

	cudaMallocManaged(&d_neibsList, neibsListSize*sizeof(*d_neibsList));
	cudaMemset(d_neibsList, -1,  neibsListSize*sizeof(*d_neibsList));

	ifstream data("data.idx");
	for (uint i = 0; i < numParticles; ++i) {
		particleinfo pi;
		data >> pi.x >> pi.y >> pi.z >> pi.w;

		h_info[i] = pi;

		data >> h_hash[i] >> h_partidx[i];
	}

	cudaMemcpy(d_info, h_info, numParticles*sizeof(*h_info), cudaMemcpyHostToDevice);
	cudaMemcpy(d_hash, h_hash, numParticles*sizeof(*h_hash), cudaMemcpyHostToDevice);

	// On GPUSPH, partidx is actually initialized on device
	initIdx<<<(numParticles + 256 - 1)/256, 256>>>(d_partidx, numParticles);
	CHECK("initIdx");

	using MyViscSpec = FullViscSpec<>;

	AbstractEngine *engine = new CUDAEngine<SPH_F1, MyViscSpec, LJ_BOUNDARY, PERIODIC_NONE, ENABLE_NONE, true>();

	engine->sort(d_info, d_hash, d_partidx, numParticles);

	cudaFree(d_partidx);
	cudaFree(d_hash);
	cudaFree(d_info);

	delete[] h_partidx;
	delete[] h_hash;
	delete[] h_info;
}
