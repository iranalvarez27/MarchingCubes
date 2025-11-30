#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <omp.h> 
#include "tablas.h"

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


void core_marching_cubes(float (*fn)(float, float, float),
                         float xmin, float ymin, float zmin,
                         float xmax, float ymax, float zmax,
                         float resolution,
                         float isoValue,
                         vector<Vector>& vertices,
                         vector<vector<int>>& faces) {

    vertices.clear(); 
    faces.clear();

    float dx = (xmax - xmin) / (resolution); 
    float dy = (ymax - ymin) / (resolution); 
    float dz = (zmax - zmin) / (resolution); 
    
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
                    int i0 = vertices.size(); vertices.push_back(edgeVerts[triTable[cubeIndex][i]]);
                    int i1 = vertices.size(); vertices.push_back(edgeVerts[triTable[cubeIndex][i+1]]);
                    int i2 = vertices.size(); vertices.push_back(edgeVerts[triTable[cubeIndex][i+2]]);

                    faces.push_back({i0, i2, i1});
                }
            }
        }
    }
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
    double Ts = end_time - start_time;
    
    write_ply(final_vertices, final_faces, filename);
    
    cout << "Tiempo de Ejecución (Ts): " << Ts << " segundos." << endl;
}


float sphereFunction(float x, float y, float z) {
    float cx=0.5f, cy=0.5f, cz=0.5f, r=0.5f;
    return (x-cx)*(x-cx)+(y-cy)*(y-cy)+(z-cz)*(z-cz)-r*r;
}


int main() {
    const int GRID_SIZE = 512; 

    
    cout << "--- Marching Cubes Secuencial (Beta 1) ---" << endl;
    cout << "Grilla de resolución: " << GRID_SIZE << "x" << GRID_SIZE << "x" << GRID_SIZE << endl;
    draw_curve(sphereFunction, "output_secuencial_512.ply", 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, GRID_SIZE); 
    
    return 0;
}