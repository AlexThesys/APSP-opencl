#pragma once

#include "apsp.h"
#include "assert.h"

#define BLOCK_SIDE 16   // the same as in kernel.cl

static cl_int move_memory_to_device(const graph_data* g_data, data_cl* cl_data);
static cl_int move_memory_to_host(const graph_data* g_data, data_cl* cl_data);
static cl_int init(data_cl* data, const char* filename);
static cl_int destroy(data_cl* data);
static cl_int setup_and_run(graph_data* g_data, data_cl* cl_data);

cl_int calculate_apsp(graph_data* g_data) {
    cl_int ret = 0;
    data_cl cl_data;
    ret = init(&cl_data, "kernel.cl");
    if (ret) {
        puts("OpenCL initialization failed.");
        return -1;
    }
    ret = setup_and_run(g_data, &cl_data);
    if (ret) {
        puts("OpenCL setup and run failed.");
        return -1;
    }
    ret = destroy(&cl_data);
    if (ret) {
        puts("OpenCL destruction.");
        return -1;
    }
    return ret;
}

static cl_int init(data_cl* data, const char* filename) {
    FILE* fp = fopen("kernel.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    char* source_str = (char*)malloc(MAX_SOURCE_SIZE);
    size_t source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1,
        &device_id, &ret_num_devices);

    // Create an OpenCL context
    data->context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

    // Create a command queue
    data->command_queue = clCreateCommandQueue(data->context, device_id, 0, &ret);  // in-order execution


    // Create a program from the kernel source
    data->program = clCreateProgramWithSource(data->context, 1,
        (const char**)&source_str, (const size_t*)&source_size, &ret);

    // Build the program
    ret = clBuildProgram(data->program, 1, &device_id, NULL, NULL, NULL);

    // Create the OpenCL kernels
    data->kernel[0] = clCreateKernel(data->program, "dependent_phase", &ret);
    data->kernel[1] = clCreateKernel(data->program, "partialy_dependent_phase", &ret);
    data->kernel[2] = clCreateKernel(data->program, "independent_phase", &ret);

    free(source_str);

    return ret;
}

static cl_int destroy(data_cl* cl_data) {
    cl_int ret = 0;
    ret = clFlush(cl_data->command_queue);
    ret = clFinish(cl_data->command_queue);
    ret = clReleaseKernel(cl_data->kernel[0]);
    ret = clReleaseKernel(cl_data->kernel[1]);
    ret = clReleaseKernel(cl_data->kernel[2]);
    ret = clReleaseProgram(cl_data->program);
    ret = clReleaseCommandQueue(cl_data->command_queue);
    ret = clReleaseContext(cl_data->context);
    return ret;
}

inline int multipleOfN(int val, int N) {
    return ((val - 1) | (N - 1)) + 1;
}

static cl_int setup_and_run(graph_data* g_data, data_cl* cl_data) {

    cl_int ret = move_memory_to_device(g_data, cl_data);
    if (ret != CL_SUCCESS) {
        puts("Failed moving memory to device.");
        return -1;
    }

    const int num_vertices = g_data->num_vertices;
    assert(num_vertices > BLOCK_SIDE);
    const int size = multipleOfN(num_vertices, BLOCK_SIDE); // make divisible by the block size
    // Execute the OpenCL kernel
    const size_t global_item_size_0[2] = { BLOCK_SIDE, BLOCK_SIDE };  // single doubly dependent block
    const size_t global_item_size_1[2] = { (size_t)size, BLOCK_SIDE * 2 };  // one row or one column
    const size_t global_item_size_2[2] = { (size_t)size, (size_t)size };  // entire workload
    const size_t local_item_size[2] = { BLOCK_SIDE, BLOCK_SIDE }; // Divide work items into groups of 64
    const int num_blocks = size / BLOCK_SIDE;

    // Set the arguments of the kernel
    for (int i = 0; i < 3; i++) {
        ret = clSetKernelArg(cl_data->kernel[i], 1, sizeof(cl_int), (void*)&num_vertices);
        ret = clSetKernelArg(cl_data->kernel[i], 2, sizeof(cl_mem), (void*)&cl_data->objects[0]);
        ret = clSetKernelArg(cl_data->kernel[i], 3, sizeof(cl_mem), (void*)&cl_data->objects[1]);
    }

    int block_id = 0;
    for (; block_id < num_blocks; block_id++) {
        // dependent phase
        ret = clSetKernelArg(cl_data->kernel[0], 0, sizeof(cl_int), (void*)&block_id);
        ret = clEnqueueNDRangeKernel(cl_data->command_queue, cl_data->kernel[0], 2, NULL, global_item_size_0, local_item_size, 0, NULL, NULL);

        // partialy dependent phase
        ret = clSetKernelArg(cl_data->kernel[1], 0, sizeof(cl_int), (void*)&block_id);
        ret = clEnqueueNDRangeKernel(cl_data->command_queue, cl_data->kernel[1], 2, NULL, global_item_size_1, local_item_size, 0, NULL, NULL);

        // independent phase
        ret = clSetKernelArg(cl_data->kernel[2], 0, sizeof(cl_int), (void*)&block_id);
        ret = clEnqueueNDRangeKernel(cl_data->command_queue, cl_data->kernel[2], 2, NULL, global_item_size_2, local_item_size, 0, NULL, NULL);
    }

    ret = move_memory_to_host(g_data, cl_data);
    if (ret != CL_SUCCESS) {
        puts("Failed moving memory to host.");
        return -1;
    }

    return ret;
}

static cl_int move_memory_to_device(const graph_data* g_data, data_cl* cl_data) {
    const int size = g_data->num_vertices * g_data->num_vertices;
    cl_int ret;

    // Create memory buffers on the device for each vector 
    cl_data->objects[0] = clCreateBuffer(cl_data->context, CL_MEM_READ_WRITE, size * sizeof(float), NULL, &ret);
    cl_data->objects[1] = clCreateBuffer(cl_data->context, CL_MEM_READ_WRITE, size * sizeof(int), NULL, &ret);

    // Copy data to their respective memory buffers
    ret = clEnqueueWriteBuffer(cl_data->command_queue, cl_data->objects[0], CL_TRUE, 0, size * sizeof(float), g_data->dist, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(cl_data->command_queue, cl_data->objects[1], CL_TRUE, 0, size * sizeof(int), g_data->path, 0, NULL, NULL);

    return ret;
}

static cl_int move_memory_to_host(const graph_data* g_data, data_cl* cl_data) {
    const int size = g_data->num_vertices * g_data->num_vertices;
    cl_int ret = 0;

    ret = clEnqueueReadBuffer(cl_data->command_queue, cl_data->objects[0], CL_TRUE, 0, size * sizeof(float), g_data->dist, 0, NULL, NULL);
    ret = clEnqueueReadBuffer(cl_data->command_queue, cl_data->objects[1], CL_TRUE, 0, size * sizeof(int), g_data->path, 0, NULL, NULL);

    ret = clReleaseMemObject(cl_data->objects[0]);
    ret = clReleaseMemObject(cl_data->objects[1]);

    return ret;
}
