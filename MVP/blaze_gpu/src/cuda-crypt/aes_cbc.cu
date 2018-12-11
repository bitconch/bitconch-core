/*
 * Copyright 2002-2016 The OpenSSL Project Authors. All Rights Reserved.
 *
 * Licensed under the OpenSSL license (the "License").  You may not use
 * this file except in compliance with the License.  You can obtain a copy
 * in the file LICENSE in the source distribution or at
 * https://www.openssl.org/source/license.html
 */

#include <algorithm>
#include "common.cu"
#include "aes.h"
#include "modes.h"
#include "perftime.h"
#include "modes_lcl.h"
#include "aes_core.cu"
#include "gpu_common.h"

#if !defined(STRICT_ALIGNMENT) && !defined(PEDANTIC)
# define STRICT_ALIGNMENT 0
#endif

__host__ __device__ void aes_cbc128_encrypt(const unsigned char* in, unsigned char* out,
                                            uint32_t len, const AES_KEY* key,
                                            unsigned char* ivec,
                                            const u32* l_te)
{
    size_t n;
    unsigned char *iv = ivec;

    if (len == 0)
        return;

#if !defined(OPENSSL_SMALL_FOOTPRINT)
    if (STRICT_ALIGNMENT &&
        ((size_t)in | (size_t)out | (size_t)ivec) % sizeof(size_t) != 0) {
        while (len >= 16) {
            for (n = 0; n < 16; ++n)
                out[n] = in[n] ^ iv[n];
            AES_encrypt(out, out, key, l_te);
            iv = out;
            len -= 16;
            in += 16;
            out += 16;
        }
    } else {
        while (len >= 16) {
            for (n = 0; n < 16; n += sizeof(size_t))
                *(size_t *)(out + n) =
                    *(size_t *)(in + n) ^ *(size_t *)(iv + n);
            AES_encrypt(out, out, key, l_te);
            iv = out;
            len -= 16;
            in += 16;
            out += 16;
        }
    }
#endif
    while (len) {
        for (n = 0; n < 16 && n < len; ++n)
            out[n] = in[n] ^ iv[n];
        for (; n < 16; ++n)
            out[n] = iv[n];
        AES_encrypt(out, out, key, l_te);
        iv = out;
        if (len <= 16)
            break;
        len -= 16;
        in += 16;
        out += 16;
    }
    memcpy(ivec, iv, 16);
}

void CRYPTO_cbc128_decrypt(const unsigned char *in, unsigned char *out,
                           size_t len, const AES_KEY *key,
                           unsigned char ivec[16], block128_f block)
{
    size_t n;
    union {
        size_t t[16 / sizeof(size_t)];
        unsigned char c[16];
    } tmp;

    if (len == 0)
        return;

#if !defined(OPENSSL_SMALL_FOOTPRINT)
    if (in != out) {
        const unsigned char *iv = ivec;

        if (STRICT_ALIGNMENT &&
            ((size_t)in | (size_t)out | (size_t)ivec) % sizeof(size_t) != 0) {
            while (len >= 16) {
                (*block) (in, out, key);
                for (n = 0; n < 16; ++n)
                    out[n] ^= iv[n];
                iv = in;
                len -= 16;
                in += 16;
                out += 16;
            }
        } else if (16 % sizeof(size_t) == 0) { /* always true */
            while (len >= 16) {
                size_t *out_t = (size_t *)out, *iv_t = (size_t *)iv;

                (*block) (in, out, key);
                for (n = 0; n < 16 / sizeof(size_t); n++)
                    out_t[n] ^= iv_t[n];
                iv = in;
                len -= 16;
                in += 16;
                out += 16;
            }
        }
        memcpy(ivec, iv, 16);
    } else {
        if (STRICT_ALIGNMENT &&
            ((size_t)in | (size_t)out | (size_t)ivec) % sizeof(size_t) != 0) {
            unsigned char c;
            while (len >= 16) {
                (*block) (in, tmp.c, key);
                for (n = 0; n < 16; ++n) {
                    c = in[n];
                    out[n] = tmp.c[n] ^ ivec[n];
                    ivec[n] = c;
                }
                len -= 16;
                in += 16;
                out += 16;
            }
        } else if (16 % sizeof(size_t) == 0) { /* always true */
            while (len >= 16) {
                size_t c, *out_t = (size_t *)out, *ivec_t = (size_t *)ivec;
                const size_t *in_t = (const size_t *)in;

                (*block) (in, tmp.c, key);
                for (n = 0; n < 16 / sizeof(size_t); n++) {
                    c = in_t[n];
                    out_t[n] = tmp.t[n] ^ ivec_t[n];
                    ivec_t[n] = c;
                }
                len -= 16;
                in += 16;
                out += 16;
            }
        }
    }
#endif
    while (len) {
        unsigned char c;
        (*block) (in, tmp.c, key);
        for (n = 0; n < 16 && n < len; ++n) {
            c = in[n];
            out[n] = tmp.c[n] ^ ivec[n];
            ivec[n] = c;
        }
        if (len <= 16) {
            for (; n < 16; ++n)
                ivec[n] = in[n];
            break;
        }
        len -= 16;
        in += 16;
        out += 16;
    }
}


void AES_cbc_encrypt(const unsigned char *in, unsigned char *out,
                     size_t len, const AES_KEY *key,
                     unsigned char *ivec, const int enc)
{

    if (enc)
        aes_cbc128_encrypt(in, out, len, key, ivec, g_Te0);
    else
        CRYPTO_cbc128_decrypt(in, out, len, key, ivec,
                              (block128_f) AES_decrypt);
}

__global__ void CRYPTO_cbc128_encrypt_kernel(const unsigned char* input, unsigned char* output,
                                             size_t length, const AES_KEY* keys,
                                             unsigned char* ivec, uint32_t num_keys,
                                             unsigned char* sha_state,
                                             uint32_t* sample_idx,
                                             uint32_t sample_len,
                                             uint32_t block_offset)
{
    size_t i = (size_t)(blockIdx.x * blockDim.x + threadIdx.x);

//#if 0
#ifdef __CUDA_ARCH__
    __shared__ u32 l_te[256];
    uint32_t tid = threadIdx.x;
    l_te[tid] = g_Te0[tid];
    __syncthreads();
#else
    const u32* l_te = g_Te0;
#endif

    if (i < num_keys) {
        aes_cbc128_encrypt(input, &output[i * length], length, &keys[i], &ivec[i * AES_BLOCK_SIZE], l_te);

        /*for (uint32_t j = 0; j < sample_len; j++) {
            if (sample_idx[j] > block_offset && sample_idx[j] < (block_offset + length)) {
            }
        }*/
    }
}

void AES_cbc_encrypt_many(const unsigned char *in, unsigned char *out,
                          size_t length, const AES_KEY *keys,
                          unsigned char *ivec,
                          uint32_t num_keys,
                          float* time_us)
{

    if (length < BLOCK_SIZE) {
        printf("ERROR! block size(%d) > length(%zu)\n", BLOCK_SIZE, length);
        return;
    }
    uint8_t* in_device = NULL;
    uint8_t* in_device0 = NULL;
    uint8_t* in_device1 = NULL;
    AES_KEY* keys_device = NULL;
    uint8_t* output_device = NULL;
    uint8_t* output_device0 = NULL;
    uint8_t* output_device1 = NULL;
    uint8_t* ivec_device = NULL;

    uint8_t* sha_state_device = NULL;

    uint32_t sample_len = 0;
    uint32_t* samples_device = NULL;

    CUDA_CHK(cudaMalloc(&in_device0, BLOCK_SIZE));
    CUDA_CHK(cudaMalloc(&in_device1, BLOCK_SIZE));

    size_t ctx_size = sizeof(AES_KEY) * num_keys;
    CUDA_CHK(cudaMalloc(&keys_device, ctx_size));
    CUDA_CHK(cudaMemcpy(keys_device, keys, ctx_size, cudaMemcpyHostToDevice));

    size_t ivec_size = AES_BLOCK_SIZE * num_keys;
    CUDA_CHK(cudaMalloc(&ivec_device, ivec_size));
    CUDA_CHK(cudaMemcpy(ivec_device, ivec, ivec_size, cudaMemcpyHostToDevice));

    size_t output_size = (size_t)num_keys * (size_t)BLOCK_SIZE;
    CUDA_CHK(cudaMalloc(&output_device0, output_size));
    CUDA_CHK(cudaMalloc(&output_device1, output_size));

    int num_threads_per_block = 256;
    int num_blocks = (num_keys + num_threads_per_block - 1) / num_threads_per_block;

    perftime_t start, end;

    get_time(&start);

    cudaStream_t stream, stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    ssize_t slength = length;
    size_t num_data_blocks = (length + BLOCK_SIZE - 1) / (BLOCK_SIZE);

    printf("num_blocks: %d threads_per_block: %d ivec_size: %zu keys size: %zu in: %p ind0: %p ind1: %p output_size: %zu num_data_blocks: %zu\n",
                    num_blocks, num_threads_per_block, ivec_size, ctx_size, in, in_device0, in_device1, output_size, num_data_blocks);

    for (uint32_t i = 0;; i++) {
        //if (i & 0x1) {
        if (0) {
            in_device = in_device1;
            output_device = output_device1;
            stream = stream1;
        } else {
            in_device = in_device0;
            output_device = output_device0;
            stream = stream0;
        }
        size_t size = std::min(slength, (ssize_t)BLOCK_SIZE);
        //printf("copying to in_device: %p in: %p size: %zu num_data_blocks: %zu\n", in_device, in, size, num_data_blocks);
        CUDA_CHK(cudaMemcpyAsync(in_device, in, size, cudaMemcpyHostToDevice, stream));

        CRYPTO_cbc128_encrypt_kernel<<<num_blocks, num_threads_per_block, 0, stream>>>(
                            in_device, output_device, size,
                            keys_device, ivec_device, num_keys,
                            sha_state_device,
                            samples_device,
                            sample_len,
                            i * BLOCK_SIZE);
#if 0
        for (uint32_t j = 0; j < num_keys; j++) {
            size_t block_offset = j * length + i * BLOCK_SIZE;
            size_t out_offset = j * size;
            //printf("i: %d j: %d copy %zi b block offset: %zu output offset: %zu num_data_blocks: %zu\n",
            //                i, j, size, block_offset, out_offset, num_data_blocks);
            CUDA_CHK(cudaMemcpy(&out[block_offset], &output_device[out_offset], size, cudaMemcpyDeviceToHost));
        }
#endif

        slength -= BLOCK_SIZE;
        in += BLOCK_SIZE;
        if (slength <= 0) {
            break;
        }
    }

    CUDA_CHK(cudaMemcpy(ivec, ivec_device, ivec_size, cudaMemcpyDeviceToHost));
    get_time(&end);
    *time_us = get_diff(&start, &end);

    //printf("gpu time: %f us\n", get_diff(&start, &end));
}


