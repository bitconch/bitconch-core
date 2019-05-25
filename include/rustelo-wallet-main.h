#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

extern void wallet_main_entry(char *network,
                              char *keypair,
                              char *timeout,
                              char *rpc_port,
                              char *proxy,
                              char *address,
                              char *airdrop_command,
                              char *airdrop_tokens,
                              char *balance_command,
                              char *cancel_command,
                              char *cancel_process_id,
                              char *confirm_command,
                              char *confirm_signature,
                              char *pay_command,
                              char *pay_to,
                              char *pay_tokens,
                              char *pay_timestamp,
                              char *pay_timestamp_pubkey,
                              char *pay_witness,
                              char *pay_cancelable,
                              char *send_signature_command,
                              char *send_signature_to,
                              char *send_signature_process_id,
                              char *send_timestamp_command,
                              char *send_timestamp_to,
                              char *send_timestamp_process_id,
                              char *send_timestamp_datetime);

#ifdef __cplusplus
}
#endif