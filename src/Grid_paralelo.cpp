#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <omp.h> 
#include "tablas.h"
#include <iomanip>

using namespace std;
struct Vector {
    float x, y, z;

    Vector(float x_=0, float y_=0, float z_=0)
      : x(x_), y(y_), z(z_) {}

    Vector operator+(const Vector& v) const {
        return Vector(x + v.x, y + v.y, z + v.z);
    }

    Vector operator-(const Vector& v) const {
        return Vector(x - v.x, y - v.y, z - v.z);
    }

    Vector operator*(float t) const {
        return Vector(x * t, y * t, z * t);
    }
};

Vector interpolate(float isoLevel,
                   const Vector& p1,
                   const Vector& p2,
                   float v1,
                   float v2) {
    if (fabs(isoLevel - v1) < 1e-6) return p1;
    if (fabs(isoLevel - v2) < 1e-6) return p2;
    if (fabs(v1 - v2) < 1e-6) return p1;

    float t = (isoLevel - v1) / (v2 - v1);
    return p1 + (p2 - p1) * t;
}


const int edgeIndex[12][2] = {
    {0,1}, {1,2}, {2,3}, {3,0},
    {4,5}, {5,6}, {6,7}, {7,4},
    {0,4}, {1,5}, {2,6}, {3,7}
};

const int vertexOffset[8][3] = {
    {0,0,0}, {1,0,0}, {1,1,0}, {0,1,0},
    {0,0,1}, {1,0,1}, {1,1,1}, {0,1,1}
};

extern const int edgeTable[256];
extern const int triTable[256][16];

void write_ply(const vector<Vector>& vertices,
               const vector<vector<int>>& faces,
               const string& filename) {
    ofstream ply(filename);
    if (!ply.is_open()) {
        cerr << "Error: could not write file " << filename << endl;
        return;
    }

    ply << "ply\n";
    ply << "format ascii 1.0\n";
    ply << "element vertex " << vertices.size() << "\n";
    ply << "property float x\n";
    ply << "property float y\n";
    ply << "property float z\n";
    ply << "element face " << faces.size() << "\n";
    ply << "property list uchar int vertex_indices\n";
    ply << "end_header\n";

    for (const auto& v : vertices)
        ply << v.x << " " << v.y << " " << v.z << "\n";

    for (const auto& f : faces)
        ply << "3 " << f[0] << " " << f[1] << " " << f[2] << "\n";

    ply.close();

    cout << "PLY written: " << filename << endl;
    cout << "Generated mesh: "
         << vertices.size() << " vertices, "
         << faces.size() << " faces.\n";
}

void core_marching_cubes(
    float (*fn)(float, float, float),
    float xmin, float ymin, float zmin,
    float xmax, float ymax, float zmax,
    int resolution,
    float isoValue,
    vector<Vector>& global_vertices,
    vector<vector<int>>& global_faces)
{
    global_vertices.clear();
    global_faces.clear();

    const float dx = (xmax - xmin) / resolution;
    const float dy = (ymax - ymin) / resolution;
    const float dz = (zmax - zmin) / resolution;

    int P = omp_get_max_threads();
    vector<vector<Vector>> local_vertices(P);
    vector<vector<vector<int>>> local_faces(P);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        #pragma omp for collapse(3) schedule(dynamic)
        for (int z = 0; z < resolution; z++) {
            for (int y = 0; y < resolution; y++) {
                for (int x = 0; x < resolution; x++) {

                    Vector cubePos[8];
                    float cubeVal[8];

                    for (int i = 0; i < 8; i++) {
                        cubePos[i] = Vector(
                            xmin + (x + vertexOffset[i][0]) * dx,
                            ymin + (y + vertexOffset[i][1]) * dy,
                            zmin + (z + vertexOffset[i][2]) * dz
                        );
                        cubeVal[i] = fn(cubePos[i].x, cubePos[i].y, cubePos[i].z);
                    }

                    int cubeIndex = 0;
                    for (int i = 0; i < 8; i++)
                        if (cubeVal[i] < isoValue) cubeIndex |= (1 << i);

                    if (edgeTable[cubeIndex] == 0) continue;

                    Vector edgeVerts[12];
                    for (int i = 0; i < 12; i++) {
                        if (edgeTable[cubeIndex] & (1 << i)) {
                            int a = edgeIndex[i][0];
                            int b = edgeIndex[i][1];
                            edgeVerts[i] = interpolate(
                                isoValue, cubePos[a], cubePos[b], cubeVal[a], cubeVal[b]
                            );
                        }
                    }

                    for (int i = 0; triTable[cubeIndex][i] != -1; i += 3) {
                        int idx0 = triTable[cubeIndex][i];
                        int idx1 = triTable[cubeIndex][i + 1];
                        int idx2 = triTable[cubeIndex][i + 2];

                        int base = local_vertices[tid].size();
                        local_vertices[tid].push_back(edgeVerts[idx0]);
                        local_vertices[tid].push_back(edgeVerts[idx1]);
                        local_vertices[tid].push_back(edgeVerts[idx2]);

                        local_faces[tid].push_back({base, base+1, base+2});
                    }
                }
            }
        }
    }

   
    int offset = 0;
    for (int t = 0; t < P; t++) {
        int base = global_vertices.size();
        global_vertices.insert(global_vertices.end(),
                               local_vertices[t].begin(),
                               local_vertices[t].end());

        for (auto& f : local_faces[t]) {
            global_faces.push_back({f[0] + base, f[1] + base, f[2] + base});
        }
    }
}


// Mover las funciones fuera de `core_marching_cubes`
float sphereFunction(float x, float y, float z) {
    float cx=0.5f, cy=0.5f, cz=0.5f, r=0.5f;
    return (x-cx)*(x-cx)+(y-cy)*(y-cy)+(z-cz)*(z-cz)-r*r;
}

void draw_curve(float (*fn)(float, float, float), 
                const string& filename, 
                float xmin, float ymin, float zmin, 
                float xmax, float ymax, float zmax, 
                int grid_resolution) {

    vector<Vector> final_vertices;
    vector<vector<int>> final_faces;
    const float ISO_LEVEL = 0.0f;
    
    double start_time = omp_get_wtime(); 

    core_marching_cubes(
        fn,
        xmin, ymin, zmin, 
        xmax, ymax, zmax, 
        (float)grid_resolution, 
        ISO_LEVEL,
        final_vertices,
        final_faces
    );
    
    double end_time = omp_get_wtime(); 
    double Tp3 = end_time - start_time;
    
    write_ply(final_vertices, final_faces, filename);
    
    cout << "Tiempo de Ejecución (Tp3 - Optimizado): " << Tp3 << " segundos." << endl;
}


#include <algorithm>  
int main() {

    using std::remove_if;   

    auto esfera = [](float x,float y,float z){  
        float cx=0.5f, cy=0.5f, cz=0.5f, r=0.35f;
        return (x-cx)*(x-cx)+(y-cy)*(y-cy)+(z-cz)*(z-cz) - r*r;
    };

    vector<int> grillas = {64,96,128,192,256,512,640,768,896,1024};
    vector<int> threads_list = {1,2,4,6,8,12,16,32,48,64,80};

    int max_threads = omp_get_max_threads();
    threads_list.erase(
        std::remove_if(threads_list.begin(), threads_list.end(),
                       [max_threads](int t){ return t > max_threads; }),
        threads_list.end()
    );

    ofstream csv("tiempos_grilla.csv");
    csv << "Grilla,Threads,Tiempo(s)\n";

    vector<Vector> global_vertices;
    vector<vector<int>> global_faces;

    for(int grilla : grillas){
        cout << "\n====== Grilla " << grilla << "³ ======\n";

        for(int num_threads : threads_list){

            omp_set_num_threads(num_threads);

            double t0 = omp_get_wtime();

            core_marching_cubes(
                esfera,    
                0.0f,0.0f,0.0f,
                1.0f,1.0f,1.0f,
                grilla,
                0.0f,
                global_vertices,
                global_faces
            );

            double t1 = omp_get_wtime();
            double tiempo = t1 - t0;

            cout << "  " << num_threads << " threads -> " << tiempo << " s\n";
            csv << grilla << "," << num_threads << "," << tiempo << "\n";
        }
    }

    csv.close();

    write_ply(global_vertices, global_faces, "resultado_grilla.ply");

    cout << "CSV generado: tiempos_grilla.csv\n";
    cout << "PLY generado: resultado_grilla.ply\n";

    return 0;
}
