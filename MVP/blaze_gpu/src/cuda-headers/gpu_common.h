#include <assert.h>

#ifndef GPU_COMMON_H
#define GPU_COMMON_H

extern bool g_verbose;

#define LOG(...) if (g_verbose) { printf(__VA_ARGS__); }

#define ROUND_UP_DIV(x, y) (((x) + (y) - 1) / (y))

#define CUDA_CHK(ans) { cuda_assert((ans), __FILE__, __LINE__); }

inline void cuda_assert(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr,"ERR: %s %s %d\n", cudaGetErrorString(err), file, line);
        assert(0);
    }
}

#endif
