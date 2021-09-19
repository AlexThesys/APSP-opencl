#include "apsp.h"

#define MAX_FSIZE 0x10000000
#define MAX_DISTANCE 100000.0f   // the same as in kernel.cl

static int read_data(const char* fname, graph_data** data);
static void print_data(graph_data* data);

int main(int argc, char** argv) {
    if (argc < 2) {
        puts("Provide filename for test values!");
        return -1;
    }
    graph_data* g_data;
    if (!!read_data(argv[1], &g_data)) {
        return -1;
    }
    if (calculate_apsp(g_data)) {
        puts("Failed calculating apsp.");
        return -1;
    }
    print_data(g_data);
    delete(g_data);

    return 0;
}

static int read_data(const char* fname, graph_data** data) {
    FILE* file = fopen(fname, "r");
    if (!file) {
        puts("Error opening file!");
        return -1;
    }
    int num_verts, num_edges;
    fscanf(file, "%d", &num_verts);
    fscanf(file, "%d", &num_edges);
    *data = new graph_data(num_verts);
    int v0, v1, dist;
    // scan all the edges
    while (EOF != fscanf(file, "%d %d %d", &v0, &v1, &dist)) {  // num_edges
        const int index = v0 * num_verts + v1;
        (*data)->dist[index] = (float)dist;
        (*data)->path[index] = v0;
    }
    // zero out edges of vertices to themself
    for (int i = 0; i < num_verts; i++) {
        const int index = i * num_verts + i;
        (*data)->dist[index] = 0.0f;
        (*data)->path[index] = -1;
    }

    fclose(file);
    return 0;
}

static void print_data(graph_data* data) {
    const int size = data->num_vertices;
    const float max_dist = (float)MAX_DISTANCE;
    puts("{\n    \"distances\":");
    printf("[");
    for (int i = 0; i < size; ++i) {
        printf("[");
        for (int j = 0; j < size; ++j) {
            if (max_dist > data->dist[i * size + j])
                printf("%.2f", data->dist[i * size + j]);
            else
                printf("-1");
            if (j != size - 1) 
                printf(",");
        }
        if (i != size - 1)
            puts("],");
        else
            printf("]");
    }
    puts("],");
    puts("    \"path\": ");
    printf("[");
    for (int i = 0; i < size; ++i) {
        printf("[");
        for (int j = 0; j < size; ++j) {
            printf("%d", data->path[i * size + j]);

            if (j != size - 1)
                printf(",");
        }
        if (i != size - 1)
            puts("],");
        else
            printf("]");
    }
    puts("],");
}
