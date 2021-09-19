#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE 0x100000
#define NUM_BUFFERS 2

struct graph_data {
    float* dist;
    int* path;
    int num_vertices;

    /* Constructor for init fields */
    graph_data(int num) : num_vertices(num) {
        const int size = num * num;
        dist = (float*)calloc(size, sizeof(float));
        path = (int*)calloc(size, sizeof(int));
    }
    ~graph_data() {
        free(path);
        free(dist);
    }
};

struct data_cl {
    cl_context context;
    cl_command_queue command_queue;
    cl_program program;
    cl_kernel kernel[3];
    cl_mem objects[2];
};

cl_int calculate_asps(graph_data* g_data);
