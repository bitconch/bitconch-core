/*
 * This file contains Solana's SGX enclave code for signing data.
 */

#include <stdbool.h>
#include <string.h>

#include "ed25519.h"
#include "signing_t.h"

static bool initialized;
static uint8_t public_key[ED25519_PUB_KEY_LEN],
    private_key[ED25519_PRIV_KEY_LEN];

/* This function creates a new public/private keypair in
   enclave trusted space.
*/
sgx_status_t init_sgx_ed25519(uint32_t keylen, uint8_t* pubkey) {
  if (keylen < sizeof(public_key)) {
    return SGX_ERROR_INVALID_PARAMETER;
  }

  uint8_t seed[ED25519_SEED_LEN];
  sgx_status_t status = sgx_read_rand(seed, sizeof(seed));
  if (SGX_SUCCESS != status) {
    return status;
  }

  ed25519_create_keypair(public_key, private_key, seed);
  memcpy(pubkey, public_key, sizeof(public_key));

  initialized = true;

  return SGX_SUCCESS;
}

/* This function signs the msg using private key.
 */
sgx_status_t sign_sgx_ed25519(uint32_t msg_len,
                              const uint8_t* msg,
                              uint32_t sig_len,
                              uint8_t* signature) {
  if (!initialized) {
    return SGX_ERROR_INVALID_STATE;
  }

  if (sig_len < ED25519_SIGNATURE_LEN) {
    return SGX_ERROR_INVALID_PARAMETER;
  }

  ed25519_sign(signature, msg, msg_len, public_key, private_key);

  return SGX_SUCCESS;
}
