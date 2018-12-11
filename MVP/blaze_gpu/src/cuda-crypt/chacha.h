#ifndef HEADER_CHACHA_H
# define HEADER_CHACHA_H

#include <inttypes.h>
# include <stddef.h>
# ifdef  __cplusplus
extern "C" {
# endif

#define CHACHA_KEY_SIZE 32
#define CHACHA_NONCE_SIZE 12
#define CHACHA_BLOCK_SIZE 64
#define CHACHA_ROUNDS 500
#define SAMPLE_SIZE 32

void __host__ __device__ chacha20_ctr_encrypt(const uint8_t *in, uint8_t *out, size_t in_len,
                                              const uint8_t key[CHACHA_KEY_SIZE], const uint8_t nonce[CHACHA_NONCE_SIZE],
                                              uint32_t counter);

void cuda_chacha20_cbc_encrypt(const uint8_t *in, uint8_t *out, size_t in_len,
                               const uint8_t key[CHACHA_KEY_SIZE], uint8_t* ivec);

void chacha_ctr_encrypt_many(const unsigned char* in, unsigned char* out,
                             size_t length,
                             const uint8_t* keys,
                             const uint8_t* nonces,
                             uint32_t num_keys,
                             float* time_us);

void chacha_cbc_encrypt_many(const uint8_t* in, uint8_t* out,
                             size_t length, const uint8_t *keys,
                             uint8_t* ivec,
                             uint32_t num_keys,
                             float* time_us);

void chacha_cbc_encrypt_many_sample(const uint8_t* in,
                                    void* out,
                                    size_t length,
                                    const uint8_t *keys,
                                    uint8_t* ivecs,
                                    uint32_t num_keys,
                                    const uint64_t* samples,
                                    uint32_t num_samples,
                                    uint64_t starting_block_offset,
                                    float* time_us);

void chacha_end_sha_state(const void* sha_state, uint8_t* out, uint32_t num_keys);

void chacha_init_sha_state(void* sha_state, uint32_t num_keys);

# ifdef  __cplusplus
}
# endif

#endif
