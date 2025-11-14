#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include "tables.h"

using namespace std;
namespace fs = std::filesystem;

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

// Linear interpolation between two points
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

// Edge vertex index pairs
const int edgeIndex[12][2] = {
  {0,1}, {1,2}, {2,3}, {3,0},
  {4,5}, {5,6}, {6,7}, {7,4},
  {0,4}, {1,5}, {2,6}, {3,7}
};

// Cube corner offsets
const int vertexOffset[8][3] = {
  {0,0,0}, {1,0,0}, {1,1,0}, {0,1,0},
  {0,0,1}, {1,0,1}, {1,1,1}, {0,1,1}
};

// Tables
extern const int edgeTable[256];
extern const int triTable[256][16];

// Write mesh to PLY file
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

// Marching Cubes Algorithm
void marching_cubes(float (*fn)(float, float, float),
                    float xmin, float ymin, float zmin,
                    float xmax, float ymax, float zmax,
                    float resolution,
                    float isoValue) {

  vector<Vector> vertices;
  vector<vector<int>> faces;

  float dx = (xmax - xmin) / (resolution - 1);
  float dy = (ymax - ymin) / (resolution - 1);
  float dz = (zmax - zmin) / (resolution - 1);

  cout << "Grid: " << resolution << "x" << resolution << "x" << resolution << endl;
  cout << "Cell size: " << dx << " x " << dy << " x " << dz << endl;

  int totalCubes = (resolution - 1) * (resolution - 1) * (resolution - 1);

  for (int z = 0; z < resolution - 1; z++) {
    for (int y = 0; y < resolution - 1; y++) {
      for (int x = 0; x < resolution - 1; x++) {

        Vector cubePos[8];
        float cubeVal[8];

        // Evaluate scalar field on cube corners
        for (int i = 0; i < 8; i++) {
          cubePos[i] = Vector(
            xmin + (x + vertexOffset[i][0]) * dx,
            ymin + (y + vertexOffset[i][1]) * dy,
            zmin + (z + vertexOffset[i][2]) * dz
          );

          cubeVal[i] = fn(cubePos[i].x, cubePos[i].y, cubePos[i].z);
        }

        // Build cube index
        int cubeIndex = 0;
        for (int i = 0; i < 8; i++)
          if (cubeVal[i] < isoValue) cubeIndex |= (1 << i);

        if (edgeTable[cubeIndex] == 0)
          continue;

        // Compute edge intersections
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

        // Generate triangles
        for (int i = 0; triTable[cubeIndex][i] != -1; i += 3) {
          int i0 = vertices.size(); vertices.push_back(edgeVerts[triTable[cubeIndex][i]]);
          int i1 = vertices.size(); vertices.push_back(edgeVerts[triTable[cubeIndex][i+1]]);
          int i2 = vertices.size(); vertices.push_back(edgeVerts[triTable[cubeIndex][i+2]]);

          faces.push_back({i0, i2, i1});
        }
      }
    }
  }

  // Write result
  write_ply(vertices, faces, "output.ply");
}

// Example implicit functions
float sphereFunction(float x, float y, float z) {
  float r = 5.0f;
  return x*x + y*y + z*z - r*r;
}

float torusFunction(float x, float y, float z) {
  float R = 5.0f;
  float r = 2.0f;
  float t = sqrt(x*x + z*z) - R;
  return t*t + y*y - r*r;
}

float cubeFunction(float x, float y, float z) {
  float s = 4.0f;
  float dx = max(fabs(x) - s, 0.0f);
  float dy = max(fabs(y) - s, 0.0f);
  float dz = max(fabs(z) - s, 0.0f);
  return dx*dx + dy*dy + dz*dz - 1.0f;
}

int main() {
  marching_cubes(
    torusFunction,
    -10, -10, -10,
     10,  10,  10,
    30,
    0.0f
  );

  return 0;
}
