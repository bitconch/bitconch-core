#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int8_t RusteloError;

/// 0 = No error.
#define RUSTELO_ERROR_SUCCESS 0

/// 1 = There was an error. Use [rustelo_last_error] to retrieve the error.
#define RUSTELO_ERROR_FAILURE 1

/// Return the error message for the last error that occurred in this SDK.
/// Error messages may only be obtained once. Further attempts will return `NULL`.
/// Returns `NULL` if no error has occurred.
extern char* rustelo_handle_error();

#ifdef __cplusplus
}
#endif
