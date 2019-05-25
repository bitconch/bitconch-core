#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

extern void benchmarker_main_entry(char *network,
                                   char *identity,
                                   char *num_nodes,
                                   char *reject_extra_node,
                                   char *threads,
                                   char *duration,
                                   char *converge_only,
                                   char *sustained,
                                   char *tx_count);

#ifdef __cplusplus
}
#endif