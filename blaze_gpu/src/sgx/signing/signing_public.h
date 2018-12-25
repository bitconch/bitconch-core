#pragma once

#include "sgx_eid.h"
#include "sgx_error.h"

#define ED25519_PUB_KEY_LEN 32

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ed25519_context {
  bool enclaveEnabled;
  sgx_enclave_id_t eid;
  uint8_t public_key[ED25519_PUB_KEY_LEN];
} ed25519_context_t;

/* This function initializes SGX enclave. It loads enclave_file
   to SGX, which internally creates a new public/private keypair.

   If the platform does not support SGX, it creates a public/private
   keypair in untrusted space. An error is returned in this scenario.
   The user can choose to not use the library if SGX encalve is not
   being used for signing.

   Note: The user must release the enclave by calling release_ed25519_context()
         after they are done using it.
*/
sgx_status_t init_ed25519(const char* enclave_file, ed25519_context_t* pctxt);

/* This function signs the msg using the internally stored private
   key. The signature is returned in the output "signature" buffer.

   This function must only be called after init_ed25519() function.
*/
sgx_status_t sign_ed25519(ed25519_context_t* pctxt,
                          uint32_t msg_len,
                          const uint8_t* msg,
                          uint32_t sig_len,
                          uint8_t* signature);

/* This function releases SGX enclave */
void release_ed25519_context(ed25519_context_t* pctxt);

#ifdef __cplusplus
}
#endif
