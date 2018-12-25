#include <stdio.h>
#include "ed25519.h"
#include <inttypes.h>
#include <assert.h>
#include <vector>
#include <pthread.h>

#define USE_CLOCK_GETTIME
#include "perftime.h"

#define LOG(...) if (verbose) { printf(__VA_ARGS__); }

#define PACKET_SIZE 512

typedef struct {
    size_t size;
    uint64_t num_retransmits;
    uint16_t addr[8];
    uint16_t port;
    bool v6;
} streamer_Meta;

typedef struct {
    uint8_t data[PACKET_SIZE];
    streamer_Meta meta;
} streamer_Packet;

bool verbose = false;

void print_dwords(unsigned char* ptr, int size) {
    for (int j = 0; j < (size)/(int)sizeof(uint32_t); j++) {
        LOG("%x ", ((uint32_t*)ptr)[j]);
    }
}

typedef struct {
    uint8_t signature[SIG_SIZE];
    uint8_t public_key[PUB_KEY_SIZE];
    uint32_t message_len;
    uint8_t message[8];
} packet_t;

typedef struct {
    gpu_Elems* elems_h;
    uint32_t num_elems;
    uint32_t message_start_offset;
    uint32_t message_len_offset;
    uint32_t signature_offset;
    uint32_t public_key_offset;
    uint8_t* out_h;
} verify_ctx_t;

static void* verify_proc(void* ctx) {
    verify_ctx_t* vctx = (verify_ctx_t*)ctx;
    LOG("sig_offset: %d pub_key_offset: %d message_start_offset: %d message_len_offset: %d\n",
        vctx->signature_offset, vctx->public_key_offset, vctx->message_start_offset, vctx->message_len_offset);
    ed25519_verify_many(&vctx->elems_h[0],
                        vctx->num_elems,
                        sizeof(streamer_Packet),
                        vctx->public_key_offset,
                        vctx->signature_offset,
                        vctx->message_start_offset,
                        vctx->message_len_offset,
                        vctx->out_h);
    return NULL;
}

int main(int argc, const char* argv[]) {
    int arg;
    for (arg = 1; arg < argc; arg++) {
        if (0 == strcmp(argv[arg], "-v")) {
            verbose = true;
        } else {
            break;
        }
    }

    if ((argc - arg) != 2) {
        printf("usage: %s [-v] <num_signatures> <num_threads>\n", argv[0]);
        return 1;
    }

    ed25519_set_verbose(verbose);

    int num_signatures = strtol(argv[arg++], NULL, 10);
    if (num_signatures <= 0) {
        printf("num_signatures should be > 0! %d\n", num_signatures);
        return 1;
    }

    int num_threads = strtol(argv[arg++], NULL, 10);
    if (num_threads <= 0) {
        printf("num_threads should be > 0! %d\n", num_signatures);
        return 1;
    }

    LOG("streamer size: %zu elems size: %zu\n", sizeof(streamer_Packet), sizeof(gpu_Elems));

    std::vector<verify_ctx_t> vctx = std::vector<verify_ctx_t>(num_threads);

    // Host allocate
    unsigned char* seed_h = (unsigned char*)calloc(num_signatures * SEED_SIZE, sizeof(uint32_t));
    unsigned char* private_key_h = (unsigned char*)calloc(num_signatures, PRIV_KEY_SIZE);
    unsigned char message_h[] = "abcd1234";
    int message_h_len = strlen((char*)message_h);
    uint32_t message_len_offset = offsetof(packet_t, message_len);
    uint32_t signature_offset = offsetof(packet_t, signature);
    uint32_t public_key_offset = offsetof(packet_t, public_key);
    uint32_t message_start_offset = offsetof(packet_t, message);

    for (int i = 0; i < num_threads; i++) {
        vctx[i].message_len_offset = message_len_offset;
        vctx[i].signature_offset = signature_offset;
        vctx[i].public_key_offset = public_key_offset;
        vctx[i].message_start_offset = message_start_offset;
    }

    std::vector<streamer_Packet> packets_h = std::vector<streamer_Packet>(num_signatures);
    int num_elems = 1;
    std::vector<gpu_Elems> elems_h = std::vector<gpu_Elems>(num_elems);
    for (int i = 0; i < num_elems; i++) {
        elems_h[i].num = num_signatures;
        elems_h[i].elems = (uint8_t*)&packets_h[0];
    }

    LOG("initing signatures..\n");
    for (int i = 0; i < num_signatures; i++) {
        packet_t* packet = (packet_t*)packets_h[i].data;
        memcpy(packet->message, message_h, message_h_len);
        packet->message_len = message_h_len + message_start_offset;

        LOG("message_len: %d sig_offset: %d pub_key_offset: %d message_start_offset: %d message_len_offset: %d\n",
            message_h_len, signature_offset, public_key_offset, message_start_offset, message_len_offset);
    }

    int out_size = num_elems * num_signatures * sizeof(uint8_t);
    for (int i = 0; i < num_threads; i++) {
        vctx[i].num_elems = num_elems;
        vctx[i].out_h = (uint8_t*)calloc(1, out_size);
        vctx[i].elems_h = &elems_h[0];
    }

    LOG("creating seed..\n");
    int ret = ed25519_create_seed(seed_h);
    LOG("create_seed: %d\n", ret);
    packet_t* first_packet_h = (packet_t*)packets_h[0].data;
    ed25519_create_keypair(first_packet_h->public_key, private_key_h, seed_h);
    ed25519_sign(first_packet_h->signature, first_packet_h->message, message_h_len, first_packet_h->public_key, private_key_h);
    ret = ed25519_verify(first_packet_h->signature, message_h, message_h_len, first_packet_h->public_key);
    LOG("verify: %d\n", ret);

    for (int i = 1; i < num_signatures; i++) {
        packet_t* packet_h = (packet_t*)packets_h[i].data;
        memcpy(packet_h->signature, first_packet_h->signature, SIG_SIZE);
        memcpy(packet_h->public_key, first_packet_h->public_key, PUB_KEY_SIZE);
    }

    for (int i = 0; i < num_signatures; i++ ) {
        packet_t* packet_h = (packet_t*)packets_h[i].data;
        unsigned char* sig_ptr = packet_h->signature;
        unsigned char* messages_ptr = packet_h->message;
        LOG("sig:");
        print_dwords(sig_ptr, SIG_SIZE);
        LOG("\nmessage: ");
        print_dwords(messages_ptr, message_h_len);
        LOG("\n\n");
    }
    LOG("\n");

    std::vector<pthread_t> threads = std::vector<pthread_t>(num_threads);
    pthread_attr_t attr;
    ret = pthread_attr_init(&attr);
    if (ret != 0) {
        LOG("ERROR: pthread_attr_init: %d\n", ret);
        return 1;
    }

    perftime_t start, end;
    get_time(&start);
    for (int i = 0; i < num_threads; i++) {
        ret = pthread_create(&threads[i],
                             &attr,
                             verify_proc,
                             &vctx[i]);
        if (ret != 0) {
            LOG("ERROR: pthread_create: %d\n", ret);
            return 1;
        }
    }

    void* res = NULL;
    for (int i = 0; i < num_threads; i++) {
        ret = pthread_join(threads[i], &res);
        if (ret != 0) {
            LOG("ERROR: pthread_join: %d\n", ret);
            return 1;
        }
    }
    get_time(&end);

    int total = (num_threads * num_signatures * num_elems);
    double diff = get_diff(&start, &end);
    printf("time diff: %f total: %d sigs/sec: %f\n",
           diff,
           total,
           (double)total / (diff / 1e6));

    for (int thread = 0; thread < num_threads; thread++) {
        LOG("ret:\n");
        bool verify_failed = false;
        for (int i = 0; i < out_size / (int)sizeof(uint8_t); i++) {
            LOG("%x ", vctx[thread].out_h[i]);
            if (vctx[thread].out_h[i] != 1) {
                verify_failed = true;
            }
        }
        LOG("\n");
        fflush(stdout);
        assert(verify_failed == false);
    }
    ed25519_free_gpu_mem();
    return 0;
}
