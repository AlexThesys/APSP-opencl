#define BLOCK_SIDE 16
#define MAX_DISTANCE 10000.0f

__kernel void dependent_phase(const int block_id, const int side_size, __global float* distances, __global int* path) {
    __local float shared_dist[BLOCK_SIDE][BLOCK_SIDE];
    __local int shared_path[BLOCK_SIDE][BLOCK_SIDE];

    const int idx = get_local_id(0);
    const int idy = get_local_id(1);

    const int v0 = BLOCK_SIDE * block_id + idy;
    const int v1 = BLOCK_SIDE * block_id + idx;

    float temp_dist;
    int temp_path;

    const int v_id = v0 * side_size + v1;
    const bool inside = (v0 < side_size) && (v1 < side_size);
    if (inside) {
        shared_dist[idy][idx] = distances[v_id];
        shared_path[idy][idx] = path[v_id];
        temp_path = shared_path[idy][idx];
    } else {
        shared_dist[idy][idx] = MAX_DISTANCE;
        shared_path[idy][idx] = -1;
    }

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

#pragma unroll
    for (int u = 0; u < BLOCK_SIDE; u++) {
        temp_dist = shared_dist[idy][u] + shared_dist[u][idx];

        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

        if (temp_path < shared_dist[idy][idx]) {
            shared_dist[idy][idx] = temp_dist;
            temp_path = shared_path[u][idx];
        }

        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

        shared_path[idy][idx] = temp_path;
    }

    if (inside) {
        distances[v_id] = shared_dist[idy][idx];
        path[v_id] = shared_path[idy][idx];
    }
}

__kernel void partialy_dependent_phase(const int block_id, const int side_size, __global float* distances, __global int* path) {
    const int block_id_x = get_group_id(0);

    if (block_id_x == block_id) 
        return;

    const int idx = get_local_id(0);
    const int idy = get_local_id(1);

    int v0 = BLOCK_SIDE * block_id + idy;
    int v1 = BLOCK_SIDE * block_id + idx;

    __local float shared_dist_base[BLOCK_SIDE][BLOCK_SIDE];
    __local int shared_path_base[BLOCK_SIDE][BLOCK_SIDE];

    // Load base block for graph and predecessors
    int v_id = v0 * side_size + v1;
    const bool inside = ((v0 < side_size) && (v1 < side_size));
    if (inside) {
        shared_dist_base[idy][idx] = distances[v_id];
        shared_path_base[idy][idx] = path[v_id];
    } else {
        shared_dist_base[idy][idx] = MAX_DISTANCE;
        shared_path_base[idy][idx] = -1;
    }

    const int block_id_y = get_group_id(1);
    // Load i-aligned singly dependent blocks
    if (block_id_y == 0) {
        v1 = BLOCK_SIDE * block_id_x + idx;
    } else {
        // Load j-aligned singly dependent blocks
        v0 = BLOCK_SIDE * block_id_x + idy;
    }

    __local float shared_dist[BLOCK_SIDE][BLOCK_SIDE];
    __local int shared_path[BLOCK_SIDE][BLOCK_SIDE];

    // Load current block for graph and predecessors
    float cur_dist;
    int cur_path;

    v_id = v0 * side_size + v1;
    if (inside) {
        cur_dist = distances[v_id];
        cur_path = path[v_id];
    } else {
        cur_dist = MAX_DISTANCE;
        cur_path = -1;
    }
    shared_dist[idy][idx] = cur_dist;
    shared_path[idy][idx] = cur_path;

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    float new_dist;
    // Compute i-aligned singly dependent blocks
    if (block_id_y == 0) {
#pragma unroll
        for (int u = 0; u < BLOCK_SIDE; u++) {
            new_dist = shared_dist_base[idy][u] + shared_dist[u][idx];

            if (new_dist < cur_dist) {
                cur_dist = new_dist;
                cur_path = shared_path[u][idx];
            }
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

            // Update new values
            shared_dist[idy][idx] = cur_dist;
            shared_path[idy][idx] = cur_path;

            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        }
    } else {
        // Compute j-aligned singly dependent blocks
#pragma unroll
        for (int u = 0; u < BLOCK_SIDE; u++) {
            new_dist = shared_dist[idy][u] + shared_dist_base[u][idx];

            if (new_dist < cur_dist) {
                cur_dist = new_dist;
                cur_path = shared_path_base[u][idx];
            }

            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

            // Update new values
            shared_dist[idy][idx] = cur_dist;
            shared_path[idy][idx] = cur_path;

            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        }
    }

    if (inside) {
        distances[v_id] = cur_dist;
        path[v_id] = cur_path;
    }
}

__kernel void independent_phase(const int block_id, const int side_size, __global float* distances, __global int* path) {
    const int block_id_x = get_group_id(0);
    const int block_id_y = get_group_id(1);

    if (block_id_x == block_id || block_id_y == block_id) 
        return;

    const int idx = get_local_id(0);
    const int idy = get_local_id(1);

    const int v0 = get_local_size(1) * block_id_y + idy;
    const int v1 = get_local_size(0) * block_id_x + idx;

    __local float shared_dist_base_row[BLOCK_SIDE][BLOCK_SIDE];
    __local float shared_dist_base_col[BLOCK_SIDE][BLOCK_SIDE];
    __local int shared_pred_base_row[BLOCK_SIDE][BLOCK_SIDE];

    const int v0_row = BLOCK_SIDE * block_id + idy;
    const int v1_col = BLOCK_SIDE * block_id + idx;

    // Load data for block
    int v_id;
    if (v0_row < side_size && v1 < side_size) {
        v_id = v0_row * side_size + v1;

        shared_dist_base_row[idy][idx] = distances[v_id];
        shared_pred_base_row[idy][idx] = path[v_id];
    } else {
        shared_dist_base_row[idy][idx] = MAX_DISTANCE;
        shared_pred_base_row[idy][idx] = -1;
    }

    if (v0 < side_size && v1_col < side_size) {
        v_id = v0 * side_size + v1_col;
        shared_dist_base_col[idy][idx] = distances[v_id];
    } else {
        shared_dist_base_col[idy][idx] = MAX_DISTANCE;
    }

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    float cur_dist;
    int cur_path;
    float new_dist;

    // Compute data for block
    if (v0 < side_size && v1 < side_size) {
        v_id = v0 * side_size + v1;
        cur_dist = distances[v_id];
        cur_path = path[v_id];

#pragma unroll
        for (int u = 0; u < BLOCK_SIDE; u++) {
            new_dist = shared_dist_base_col[idy][u] + shared_dist_base_row[u][idx];
            if (cur_dist > new_dist) {
                cur_dist = new_dist;
                cur_path = shared_pred_base_row[u][idx];
            }
        }
        distances[v_id] = cur_dist;
        path[v_id] = cur_path;
    }
}