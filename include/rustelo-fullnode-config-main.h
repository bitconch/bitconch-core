#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

extern void fullnode_config_main_entry(char *local,
                                       char *keypair,
                                       char *public,
                                       char *bind,
                                       char *outfile);

#ifdef __cplusplus
}
#endif