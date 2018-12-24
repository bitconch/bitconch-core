#include "sha512.h"
#include <algorithm>
#include <stdio.h>
#include "ge.cu"
#include "sc.cu"
#include "fe.cu"
#include "seed.cu"
#include "keypair.cu"
#include "sign.cu"
#include "sha512.cu"

#include "ed25519.h"
#include <pthread.h>

#include "gpu_common.h"

#define USE_CLOCK_GETTIME
#include "perftime.h"

static int __host__ __device__ consttime_equal(const unsigned char *x, const unsigned char *y) {
    unsigned char r = 0;

    r = x[0] ^ y[0];
    #define F(i) r |= x[i] ^ y[i]
    F(1);
    F(2);
    F(3);
    F(4);
    F(5);
    F(6);
    F(7);
    F(8);
    F(9);
    F(10);
    F(11);
    F(12);
    F(13);
    F(14);
    F(15);
    F(16);
    F(17);
    F(18);
    F(19);
    F(20);
    F(21);
    F(22);
    F(23);
    F(24);
    F(25);
    F(26);
    F(27);
    F(28);
    F(29);
    F(30);
    F(31);
    #undef F

    return !r;
}

static int __device__ __host__
ed25519_verify_device(const unsigned char *signature,
                      const unsigned char *message,
                      uint32_t message_len,
                      const unsigned char *public_key) {
    unsigned char h[64];
    unsigned char checker[32];
    sha512_context hash;
    ge_p3 A;
    ge_p2 R;

    if (signature[63] & 224) {
        return 0;
    }

    if (ge_frombytes_negate_vartime(&A, public_key) != 0) {
        return 0;
    }

    sha512_init(&hash);
    sha512_update(&hash, signature, 32);
    sha512_update(&hash, public_key, 32);
    sha512_update(&hash, message, message_len);
    sha512_final(&hash, h);
    
    sc_reduce(h);
    ge_double_scalarmult_vartime(&R, h, &A, signature + 32);
    ge_tobytes(checker, &R);

    if (!consttime_equal(checker, signature)) {
        return 0;
    }

    return 1;
}

int 
ed25519_verify(const unsigned char *signature,
               const unsigned char *message,
               uint32_t message_len,
               const unsigned char *public_key) {
    return ed25519_verify_device(signature, message, message_len, public_key);
}

__global__ void ed25519_verify_kernel(const uint8_t* packets,
                                      uint32_t message_size,
                                      uint32_t public_key_offset,
                                      uint32_t signature_offset,
                                      uint32_t message_start_offset,
                                      uint32_t message_len_offset,
                                      size_t num_keys,
                                      uint8_t* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_keys) {
        const uint8_t* packet = &packets[i * message_size];
        uint32_t message_len = ((uint32_t*)&packet[message_len_offset])[0];
        if (message_len > message_start_offset) {
            out[i] = ed25519_verify_device(&packet[signature_offset],
                                           &packet[message_start_offset],
                                           message_len - message_start_offset,
                                           &packet[public_key_offset]);
        }
    }
}

typedef struct {
    uint8_t* packets;
    uint8_t* out;

    size_t num;
    uint32_t total_packets_len;
    pthread_mutex_t mutex;
} gpu_ctx;

static pthread_mutex_t g_ctx_mutex = PTHREAD_MUTEX_INITIALIZER;

#define MAX_NUM_GPUS 8
#define MAX_QUEUE_SIZE 8

static gpu_ctx g_gpu_ctx[MAX_NUM_GPUS][MAX_QUEUE_SIZE] = {0};
static uint32_t g_cur_gpu = 0;
static uint32_t g_cur_queue[MAX_NUM_GPUS] = {0};
static int32_t g_total_gpus = -1;
bool g_verbose = false;

void ed25519_set_verbose(bool val) {
    g_verbose = val;
}

static bool ed25519_init_locked() {
    if (g_total_gpus == -1) {
        cudaGetDeviceCount(&g_total_gpus);
        g_total_gpus = min(8, g_total_gpus);
        LOG("total_gpus: %d\n", g_total_gpus);
        for (int gpu = 0; gpu < g_total_gpus; gpu++) {
            for (int queue = 0; queue < MAX_QUEUE_SIZE; queue++) {
                int err = pthread_mutex_init(&g_gpu_ctx[gpu][queue].mutex, NULL);
                if (err != 0) {
                    fprintf(stderr, "pthread_mutex_init error %d gpu: %d queue: %d\n",
                            err, gpu, queue);
                    g_total_gpus = 0;
                    return false;
                }
            }
        }
    }
    return g_total_gpus > 0;
}

bool ed25519_init() {
    cudaFree(0);
    pthread_mutex_lock(&g_ctx_mutex);
    bool success = ed25519_init_locked();
    pthread_mutex_unlock(&g_ctx_mutex);
    return success;
}

void ed25519_verify_many(const gpu_Elems* elems,
                         uint32_t num,
                         uint32_t message_size,
                         uint32_t public_key_offset,
                         uint32_t signature_offset,
                         uint32_t message_start_offset,
                         uint32_t message_len_offset,
                         uint8_t* out)
{
    size_t out_size = 0;
    LOG("Starting verify_many keys: %d\n message_size: %d message_start_offset: %d\n",
        (int)num, message_size, message_start_offset);

    uint32_t total_packets_len = 0;
    uint32_t total_packets = 0;

    for (size_t i = 0; i < num; i++) {
        total_packets += elems[i].num;
        total_packets_len += elems[i].num * message_size;
        out_size += elems[i].num * sizeof(uint8_t);
    }

    if (0 == total_packets) {
        return;
    }

    int32_t cur_gpu, cur_queue;

    LOG("device allocate. packets: %d out: %d\n", total_packets_len, (int)out_size);
    // Device allocate

    pthread_mutex_lock(&g_ctx_mutex);
    if (!ed25519_init_locked()) {
        pthread_mutex_unlock(&g_ctx_mutex);
        LOG("No GPUs, exiting...\n");
        return;
    }
    cur_gpu = g_cur_gpu;
    g_cur_gpu++;
    g_cur_gpu %= g_total_gpus;
    cur_queue = g_cur_queue[cur_gpu];
    g_cur_queue[cur_gpu]++;
    g_cur_queue[cur_gpu] %= MAX_QUEUE_SIZE;
    pthread_mutex_unlock(&g_ctx_mutex);

    gpu_ctx* cur_ctx = &g_gpu_ctx[cur_gpu][cur_queue];
    pthread_mutex_lock(&cur_ctx->mutex);

    cudaSetDevice(cur_gpu);

    LOG("cur gpu: %d cur queue: %d\n", cur_gpu, cur_queue);

    if (cur_ctx->packets == NULL ||
        total_packets_len > cur_ctx->total_packets_len) {
        CUDA_CHK(cudaFree(cur_ctx->packets));
        CUDA_CHK(cudaMalloc(&cur_ctx->packets, total_packets_len));

        cur_ctx->total_packets_len = total_packets_len;
    }

    if (cur_ctx->out == NULL || cur_ctx->num < total_packets) {
        CUDA_CHK(cudaFree(cur_ctx->out));
        CUDA_CHK(cudaMalloc(&cur_ctx->out, out_size));

        cur_ctx->num = total_packets;
    }

    size_t cur = 0;
    for (size_t i = 0; i < num; i++) {
        LOG("i: %zu size: %d\n", i, elems[i].num * message_size);
        CUDA_CHK(cudaMemcpy(&cur_ctx->packets[cur * message_size], elems[i].elems, elems[i].num * message_size, cudaMemcpyHostToDevice));
        cur += elems[i].num;

        if (elems[i].num > 0) {
            uint32_t message_len = ((uint32_t*)&elems[i].elems[message_len_offset])[0];
            LOG("message_len: %d\n", message_len);

            for (size_t j = 0; j < message_len; j++) {
                LOG("%d ", elems[i].elems[message_start_offset + j]);
            }
            LOG("\n");
        }
    }

    int num_threads_per_block = 64;
    int num_blocks = ROUND_UP_DIV(total_packets, num_threads_per_block);
    LOG("num_blocks: %d threads_per_block: %d keys: %d out: %p\n",
           num_blocks, num_threads_per_block, (int)total_packets, out);

    if (num > 0 && elems[0].num > 0) {
        LOG("signature: ");
        for (int i = 0; i < SIG_SIZE; i++) {
            LOG("%d ", elems[0].elems[signature_offset + i]);
        }
        LOG("\n");

        LOG("pub_key: ");
        for (int i = 0; i < PUB_KEY_SIZE; i++) {
            LOG("%d ", elems[0].elems[public_key_offset + i]);
        }
        LOG("\n");
    }

    perftime_t start, end;
    get_time(&start);
    ed25519_verify_kernel<<<num_blocks, num_threads_per_block>>>
                            (cur_ctx->packets,
                             message_size,
                             public_key_offset,
                             signature_offset,
                             message_start_offset,
                             message_len_offset,
                             cur_ctx->num,
                             cur_ctx->out);
    CUDA_CHK(cudaPeekAtLastError());

    cudaError_t err = cudaMemcpy(out, cur_ctx->out, out_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)  {
        fprintf(stderr, "cudaMemcpy(out) error: out = %p cur_ctx->out = %p size = %zu num: %d elems = %p\n",
                        out, cur_ctx->out, out_size, num, elems);
    }
    CUDA_CHK(err);

    pthread_mutex_unlock(&cur_ctx->mutex);

    get_time(&end);
    LOG("time diff: %f\n", get_diff(&start, &end));
}

void ed25519_free_gpu_mem() {
    for (size_t gpu = 0; gpu < MAX_NUM_GPUS; gpu++) {
        for (size_t queue = 0; queue < MAX_QUEUE_SIZE; queue++) {
            gpu_ctx* cur_ctx = &g_gpu_ctx[gpu][queue];
            CUDA_CHK(cudaFree(cur_ctx->packets));
            CUDA_CHK(cudaFree(cur_ctx->out));
        }
    }
}

// Ensure copyright and license notice is embedded in the binary
const char* ed25519_license() {
   return "Copyright (c) 2018 Solana Labs, Inc. "
          "Licensed under the Apache License, Version 2.0 "
          "<http://www.apache.org/licenses/LICENSE-2.0>";
}
