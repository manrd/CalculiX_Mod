#ifndef CCX_PROF_H
#define CCX_PROF_H

#include <time.h>
#include <stdio.h>
#include <stdlib.h>

static inline double ccx_prof_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec * 1e-6;
}

static inline int ccx_prof_enabled(void) {
    static int init = 0;
    static int enabled = 0;
    if (!init) {
        const char *e = getenv("CCX_PROFILE");
        enabled = (e && e[0] != '\0' && e[0] != '0') ? 1 : 0;
        init = 1;
    }
    return enabled;
}

#define CCX_PROF_LOG(fmt, ...) \
    do { if (ccx_prof_enabled()) { printf("[PROF] " fmt "\n", ##__VA_ARGS__); fflush(stdout); } } while(0)

#endif
