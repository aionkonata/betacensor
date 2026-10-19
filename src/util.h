#ifndef BC_UTIL_H
#define BC_UTIL_H

#include <stdio.h>
#include <stdlib.h>

#define LOG(fmt, ...)  fprintf(stderr, "[bc] " fmt "\n", ##__VA_ARGS__)
#define WARN(fmt, ...) fprintf(stderr, "[bc] WARN: " fmt "\n", ##__VA_ARGS__)
#define ERR(fmt, ...)  fprintf(stderr, "[bc] ERROR: " fmt "\n", ##__VA_ARGS__)

#define DIE(fmt, ...) do { ERR(fmt, ##__VA_ARGS__); exit(1); } while (0)

#endif /* BC_UTIL_H */
