#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include "tables.h"
#define DEBUG(i) printf("CHECKPOINT %f\n", i)
#define NUMVAR(key, val) printf("%s: %f\n", key, val)
#define NUMVAL(exp, val) printf("%s: %f\n", exp, val)

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

// Auxiliary Function eval_center
float eval_center(float x0, float y0, float z0,
                  float x1, float y1, float z1,
                  float (*fn)(float, float, float)) {

  float cx = (x0 + x1)*0.5f;
  float cy = (y0 + y1)*0.5f;
  float cz = (z0 + z1)*0.5f;
  return fn(cx, cy, cz);
}

// Auxiliary Function empty_block
bool empty_block(float value_at_center,
                 float isoValue,
                 float edge_length) {

  float threshold = isoValue + (sqrt(3.0f)/2.0f)*edge_length;
  return value_at_center > threshold;
}

// Auxiliary Function process_single_cube
void process_single_cube(float (*fn)(float, float, float),
                         float x0, float y0, float z0,
                         float x1, float y1, float z1,
                         float isoValue,
                         vector<Vector> &vertices,
                         vector<vector<int>> &faces) {

  // Compute the 8 corners in real coords
  Vector corner[8] = {
    {x0, y0, z0},
    {x1, y0, z0},
    {x1, y1, z0},
    {x0, y1, z0},
    {x0, y0, z1},
    {x1, y0, z1},
    {x1, y1, z1},
    {x0, y1, z1}
  };

  // Evaluate the scalar field at each corner
  float value[8];
  for (int i = 0; i < 8; i++)
    value[i] = fn(corner[i].x, corner[i].y, corner[i].z);

  // Create the cube index
  int cubeIndex = 0;
  for (int i = 0; i < 8; i++)
    if (value[i] < isoValue)
      cubeIndex |= (1 << i);

  // Cube is entirely outside or inside → no triangles
  if (edgeTable[cubeIndex] == 0)
    return;

  // Compute intersection vertices on edges
  Vector edgeVertex[12];

  for (int e = 0; e < 12; e++) {
    if (edgeTable[cubeIndex] & (1 << e)) {
      int v0 = edgeIndex[e][0];
      int v1 = edgeIndex[e][1];
      edgeVertex[e] = interpolate(isoValue,
                                  corner[v0], corner[v1],
                                  value[v0], value[v1]);
    }
  }

  // Create triangles
  for (int t = 0; triTable[cubeIndex][t] != -1; t += 3) {
    int idx0 = triTable[cubeIndex][t];
    int idx1 = triTable[cubeIndex][t + 1];
    int idx2 = triTable[cubeIndex][t + 2];

    int a = vertices.size();
    vertices.push_back(edgeVertex[idx0]);
    int b = vertices.size();
    vertices.push_back(edgeVertex[idx1]);
    int c = vertices.size();
    vertices.push_back(edgeVertex[idx2]);

    //faces.push_back({a, b, c});
    faces.push_back({a, c, b});
  }
}

// Auxiliary Function process_octree
void process_octree(float (*fn)(float, float, float),
                    float x0, float y0, float z0,
                    float x1, float y1, float z1,
                    int depth,
                    float isoValue,
                    vector<Vector> &vertices,
                    vector<vector<int>> &faces,
                    float dx, float dy, float dz) {
  
  // If we are at the lowest level, process with marching cubes
  //DEBUG(1.1);
  if (depth == 0) {
    process_single_cube(fn, x0, y0, z0, x1, y1, z1,
                        isoValue, vertices, faces);
    return;
  }

  // Evaluate at the 8 corners
  float f000 = fn(x0, y0, z0);
  float f100 = fn(x1, y0, z0);
  float f010 = fn(x0, y1, z0);
  float f110 = fn(x1, y1, z0);
  float f001 = fn(x0, y0, z1);
  float f101 = fn(x1, y0, z1);
  float f011 = fn(x0, y1, z1);
  float f111 = fn(x1, y1, z1);

  // Find the minimum and maximum of the corners
  float fmin = f000;
  float fmax = f000;
  
  fmin = min(fmin, f100); fmax = max(fmax, f100);
  fmin = min(fmin, f010); fmax = max(fmax, f010);
  fmin = min(fmin, f110); fmax = max(fmax, f110);
  fmin = min(fmin, f001); fmax = max(fmax, f001);
  fmin = min(fmin, f101); fmax = max(fmax, f101);
  fmin = min(fmin, f011); fmax = max(fmax, f011);
  fmin = min(fmin, f111); fmax = max(fmax, f111);

  // Only prune if isoValue is completely outside the range [fmin, fmax]
  if (depth <= 2 && (isoValue < fmin || isoValue > fmax)) {
    //NUMVAL("fmin", fmin);
    //NUMVAL("fmax", fmax);
    //NUMVAL("isoValue", isoValue);
    //DEBUG(1.2);
    return; // Prune this node
  }

  // Subdivide into 8 children
  float mx = (x0 + x1) * 0.5f;
  float my = (y0 + y1) * 0.5f;
  float mz = (z0 + z1) * 0.5f;

  process_octree(fn, x0, y0, z0, mx, my, mz, depth-1, isoValue, vertices, faces, dx, dy, dz);
  process_octree(fn, mx, y0, z0, x1, my, mz, depth-1, isoValue, vertices, faces, dx, dy, dz);
  process_octree(fn, x0, my, z0, mx, y1, mz, depth-1, isoValue, vertices, faces, dx, dy, dz);
  process_octree(fn, mx, my, z0, x1, y1, mz, depth-1, isoValue, vertices, faces, dx, dy, dz);
  
  process_octree(fn, x0, y0, mz, mx, my, z1, depth-1, isoValue, vertices, faces, dx, dy, dz);
  process_octree(fn, mx, y0, mz, x1, my, z1, depth-1, isoValue, vertices, faces, dx, dy, dz);
  process_octree(fn, x0, my, mz, mx, y1, z1, depth-1, isoValue, vertices, faces, dx, dy, dz);
  process_octree(fn, mx, my, mz, x1, y1, z1, depth-1, isoValue, vertices, faces, dx, dy, dz);
}

// Marching Cubes Octree-Mesh Algorithm
void marching_cubes_octree(float (*fn)(float, float, float),
                          float xmin, float ymin, float zmin,
                          float xmax, float ymax, float zmax,
                          int resolution,
                          float isoValue) {

  vector<Vector> vertices;
  vector<vector<int>> faces;

  // Compute voxel sizes
  float dx = (xmax - xmin) / resolution;
  float dy = (ymax - ymin) / resolution;
  float dz = (zmax - zmin) / resolution;

  // Depth of octree = log2(resolution)
  int depth = 0;
  int tmp = resolution;
  while (tmp > 1) { tmp >>= 1; depth++; }
  
  //DEBUG(1);
  // Start octree on the whole domain
  process_octree(fn, xmin, ymin, zmin, xmax, ymax, zmax,
    depth, isoValue, vertices, faces, dx, dy, dz);
    
  //DEBUG(2);
  // Output mesh
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
  marching_cubes_octree(
    sphereFunction,
    -7, -7, -7,   // x,y,z min
     7,  7,  7,   // x,y,z max
    32,           // resolution must be a power of two
    0.0f
  );

  /*
  marching_cubes_octree(
    torusFunction,
    -7, -2, -7,   // x,y,z min
    7,  2,  7,    // x,y,z max
    32,           // resolution must be a power of two
    0.0f
  );
  */

  return 0;
}
