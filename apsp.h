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
    float* dist;    // minimum distance
    int* path;  // vertex - predecessor on the path
    int num_vertices;

    /* Constructor for init fields */
    void init(int num) {
        num_vertices = num;
        const int size = num * num;
        dist = (float*)malloc(size * sizeof(float));
        path = (int*)malloc(size * sizeof(int));
    }
    ~graph_data() {
        free(path);
        free(dist);
    }
};

struct apsp_cl {
    cl_context context;
    cl_command_queue command_queue;
    cl_program program;
    cl_kernel kernel[3];
    cl_mem objects[2];
    cl_device_id device_id = 0;

    cl_int move_memory_to_device(const graph_data* g_data);
    cl_int move_memory_to_host(const graph_data* g_data);
public:
    cl_int init(const char* filename);
    cl_int destroy();
    cl_int setup_and_run(graph_data* g_data);
};

// call this to compute apsp
cl_int calculate_apsp(graph_data* g_data);
