/*
 * This file contains Solana's SGX enclave code for signing data.
 */

#include <stdbool.h>
#include <string.h>
#include <time.h>

#include "ed25519.h"
#include "sgx_urts.h"
#include "signing_public.h"
#include "signing_u.h"

/* This function initializes SGX enclave. It loads enclave_file
   to SGX, which internally creates a new public/private keypair.
*/
sgx_status_t init_ed25519(const char* enclave_file, ed25519_context_t* pctxt) {
  int updated = 0;
  sgx_launch_token_t token = {0};
  sgx_enclave_id_t eid;

  // Try to load the SGX enclave
  sgx_status_t status =
      sgx_create_enclave(enclave_file, 1, &token, &updated, &eid, NULL);

  if (SGX_SUCCESS != status) {
    return status;
  }

  sgx_status_t retval = SGX_SUCCESS;
  status = init_sgx_ed25519(eid, &retval, sizeof(pctxt->public_key),
                            &pctxt->public_key[0]);

  if (SGX_SUCCESS != status) {
    sgx_destroy_enclave(eid);
    return status;
  }

  if (SGX_SUCCESS != retval) {
    sgx_destroy_enclave(eid);
    return retval;
  }

  pctxt->enclaveEnabled = true;
  pctxt->eid = eid;

  return status;
}

/* This function signs the msg using the internally stored private
   key. The signature is returned in the output "signature" buffer.

   This function must only be called after init_ed25519() function.
*/
sgx_status_t sign_ed25519(ed25519_context_t* pctxt,
                          uint32_t msg_len,
                          const uint8_t* msg,
                          uint32_t sig_len,
                          uint8_t* signature) {
  if (!pctxt->enclaveEnabled) {
    return SGX_ERROR_INVALID_ENCLAVE;
  }

  sgx_status_t retval = SGX_SUCCESS;
  sgx_status_t status =
      sign_sgx_ed25519(pctxt->eid, &retval, msg_len, msg, sig_len, signature);

  if (SGX_SUCCESS != status) {
    return status;
  }

  if (SGX_SUCCESS != retval) {
    return retval;
  }

  return status;
}

void release_ed25519_context(ed25519_context_t* pctxt) {
  sgx_destroy_enclave(pctxt->eid);
}