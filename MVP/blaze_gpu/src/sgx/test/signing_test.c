#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include "signing_public.h"

#include "ed25519.h"

void print_buffer(const uint8_t* buf, int len) {
  char str[BUFSIZ] = {'\0'};
  int offset = 0;
  for (int i = 0; i < len; i++) {
    offset += snprintf(&str[offset], BUFSIZ - offset, "0x%02x ", buf[i]);
    if (!((i + 1) % 8))
      offset += snprintf(&str[offset], BUFSIZ - offset, "\n");
  }
  offset += snprintf(&str[offset], BUFSIZ - offset, "\n");
  printf("%s", str);
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    printf("Usage: %s <enclave file path>\n", argv[0]);
    return -1;
  }

  ed25519_context_t ctxt;
  sgx_status_t status = init_ed25519(argv[1], &ctxt);
  if (SGX_SUCCESS != status) {
    printf("Failed in init_ed25519. Error %d\n", status);
    return -1;
  }

  printf("Loaded the enclave. eid: %d\n", (uint32_t)ctxt.eid);

  uint8_t* data =
      "This is a test string. We'll sign it using SGX enclave. Hope it works!!";
  uint8_t signature[64];
  memset(signature, 0, sizeof(signature));
  status =
      sign_ed25519(&ctxt, sizeof(data), data, sizeof(signature), signature);
  if (SGX_SUCCESS != status) {
    printf("Failed in sign_ed25519. Error %d\n", status);
    release_ed25519_context(&ctxt);
    return -1;
  }

  printf("Signature:\n");
  print_buffer(signature, sizeof(signature));

  if (ed25519_verify(signature, data, sizeof(data), ctxt.public_key) == 0) {
     printf("Failed in verifying the signature\n");
  } else {
     printf("Signature verified\n");
  }

  release_ed25519_context(&ctxt);
  return 0;
}