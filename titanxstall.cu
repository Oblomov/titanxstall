/// Copyright (C) 2016 Giuseppe Bilotta <giuseppe.bilotta@gmail.com>
/// License: GPLv3

#include <sstream>
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

#include "cached_alloc.h"

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

static cached_allocator *cacher;

typedef void (*sort_func_ptr)(key_iterator const&, key_iterator const&, thrust_uint_ptr const&);

void default_sort(key_iterator const& key_start, key_iterator const& key_end, thrust_uint_ptr const& val_start)
{
	ptype_hash_compare comp;

	thrust::sort_by_key(key_start, key_end, val_start, comp);
}

void caching_sort(key_iterator const& key_start, key_iterator const& key_end, thrust_uint_ptr const& val_start)
{
	ptype_hash_compare comp;

	thrust::sort_by_key(thrust::cuda::par(*cacher), key_start, key_end, val_start, comp);
}

static sort_func_ptr sort_func = &default_sort;


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

	sort_func(key_start, key_end, particleIndex);
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

using namespace std;

static string info_name;
static FILE *infostream;

void timestamp(string const& msg)
{
	static long int init_pos = 0;

	char timestamp[36] = {0};
	time_t t;
	struct tm *tmp;
	size_t wrt = 0;
	t = time(NULL);
	tmp = localtime(&t);
	if (tmp) {
		wrt = strftime(timestamp, 36, "[%Y-%m-%dT%H:%M:%S] ", tmp);
	}
	if (!wrt) {
		snprintf(timestamp, 36, "[unknown]");
	}
	fprintf(infostream, "%s: %s\n", timestamp, msg.c_str());
	fflush(infostream);
	if (init_pos)
		fseek(infostream, init_pos, SEEK_SET);
	else
		init_pos = ftell(infostream);
}

// output the content of the scratch stringstream to the infostream, time-stamped,
// and optionally to the console too
// clear the scratch stringstream afterwards
void report(stringstream &scratch, bool on_console=false)
{
	timestamp(scratch.str());
	if (on_console)
		cout << scratch.str() << endl;
	scratch.str("");
}

void cleanup(void)
{
	if (infostream) {
		fclose(infostream);
		shm_unlink(info_name.c_str());
		infostream = NULL;
	}
	if (cacher) {
		delete cacher;
		cacher = NULL;
	}
	cudaDeviceReset();
}

void sig_handler(int signum)
{
	cleanup();
	exit(1);
}

void check(const char *file, unsigned long line, const char *func)
{
	stringstream errmsg;
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		errmsg << file << ":" << line << " in " << func
			<< ": runtime API error " << err << " (" << cudaGetErrorString(err) << ")";
		throw runtime_error(errmsg.str());
	}
}


int main(int argc, char *argv[])
{
	stringstream scratch;

	scratch << "/titanxfall-" << getpid();

	info_name = scratch.str();
	scratch.str("");

	int ret = shm_open(info_name.c_str(), O_RDWR | O_CREAT, S_IRWXU);
	if (ret < 0)
		throw runtime_error("can't open info stream");

	infostream = fdopen(ret, "w");
	if (!infostream)
		throw runtime_error("can't fdopen info stream");


	// catch SIGINT
	struct sigaction int_action;
	memset(&int_action, 0, sizeof(int_action));
	int_action.sa_handler = sig_handler;
	ret = sigaction(SIGINT, &int_action, NULL);

	if (ret < 0)
		throw runtime_error("can't register info stream cleanup function");

	uint numParticles = 5*1024*1024;
	uint device = 0;
	bool custom_alloc = false;

	const char * const * arg = argv + 1;
	while (argc > 1) {
		if (!strcmp(*arg, "--device")) {
			if (argc < 2)
				throw invalid_argument("please specify a device");
			--argc;
			++arg;
			device = atoi(*arg);
		} else if (!strcmp(*arg, "--elements")) {
			if (argc < 2)
				throw invalid_argument("please specify a device");
			--argc;
			++arg;
			numParticles = atoi(*arg);
		} else if (!strcmp(*arg, "--cache-alloc")) {
			custom_alloc = true;
		}
		--argc;
		++arg;

	}

	cudaSetDevice(device);
	CHECK("set device");

	scratch << "Initializing PID " << getpid() << " device " << device << " particles " << numParticles << " ...";
	report(scratch, true);

	if (custom_alloc) {
		cacher = new cached_allocator;
		sort_func = &caching_sort;

		scratch << "Caching allocator enabled";
		report(scratch, true);
	}
	unsigned long counter = 0;

	particleinfo *info = NULL;
	hashKey *hash = NULL;
	uint *partidx = NULL;

	cudaMalloc(&info, numParticles*sizeof(*info));
	cudaMalloc(&hash, numParticles*sizeof(*hash));
	cudaMalloc(&partidx, numParticles*sizeof(*partidx));

	const int blockSize = 1024;
	const int numBlocks = (numParticles + blockSize - 1)/blockSize;

	initParticles<<<numBlocks, blockSize>>>(info, hash, partidx, numParticles);
	CHECK("initParticles");

	while (true) {
		reHashParticles<<<numBlocks, blockSize>>>(info, hash, partidx, numParticles);
		CHECK("reHashParticles");

		sort(info, hash, partidx, numParticles);
		CHECK("sort");

		++counter;

		scratch << "Iteration " << counter;

		report(scratch, counter % 1000 == 0);
	}
}

