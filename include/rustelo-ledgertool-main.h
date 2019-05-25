#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

extern void ledgertool_main_entry(char *ledger,
                                  char *head,
                                  char *precheck,
                                  char *continu,
                                  char *subcommand);

#ifdef __cplusplus
}
#endif