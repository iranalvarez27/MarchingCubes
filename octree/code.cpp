#include <iostream>
#include <fstream>
#include <vector>
#include <functional>
#include <cmath>
#include "tablas.h"

using namespace std;

// ------------------------------
// Vector básico
// ------------------------------
struct Vector {
    double x, y, z;

    Vector(double x_=0, double y_=0, double z_=0)
        : x(x_), y(y_), z(z_) {}

    Vector operator+(const Vector& v) const {
        return Vector(x+v.x, y+v.y, z+v.z);
    }

    Vector operator-(const Vector& v) const {
        return Vector(x-v.x, y-v.y, z-v.z);
    }

    Vector operator*(double t) const {
        return Vector(x*t, y*t, z*t);
    }
};

// -----------------------------------
// Definiciones Marching Cubes básicas
// -----------------------------------
extern const int tablaDeAristas[256];
extern const int triTable[256][16];

const int aristaIndice[12][2] = {
    {0,1},{1,2},{2,3},{3,0},
    {4,5},{5,6},{6,7},{7,4},
    {0,4},{1,5},{2,6},{3,7}
};

const Vector vertCubo[8] = {
    Vector(0,0,0), Vector(1,0,0), Vector(1,1,0), Vector(0,1,0),
    Vector(0,0,1), Vector(1,0,1), Vector(1,1,1), Vector(0,1,1)
};

// -----------------------------------
// Interpolación estándar Marching Cubes
// -----------------------------------
Vector interp(double iso, const Vector& p1, const Vector& p2, double v1, double v2) {
    if (fabs(iso - v1) < 1e-6) return p1;
    if (fabs(iso - v2) < 1e-6) return p2;

    if (fabs(v1 - v2) < 1e-12) return p1;

    double t = (iso - v1) / (v2 - v1);
    return p1 + (p2 - p1) * t;
}

// ------------------------------
// Vectores globales
// ------------------------------
vector<Vector> vertices;
vector<vector<int>> caras;

// ----------------------------------------------------------
// Procesar celda cúbica — EXACTO A TU MARCHING CUBES DE GRILLA
// ----------------------------------------------------------
void procesarCelda(
    function<double(double,double,double)> f,
    const Vector& minCorner,
    double size,
    double iso
) {
    Vector pos[8];
    double val[8];

    // Esquinas del cubo
    for (int i=0; i<8; i++) {
        pos[i] = minCorner + vertCubo[i] * size;
        val[i] = f(pos[i].x, pos[i].y, pos[i].z);
    }

    // Identificador MC
    int cubeIndex = 0;
    for (int i=0; i<8; i++)
        if (val[i] < iso) cubeIndex |= (1 << i);

    if (tablaDeAristas[cubeIndex] == 0) return;

    // Intersecciones
    Vector edgeVert[12];
    for (int i=0; i<12; i++) {
        if (tablaDeAristas[cubeIndex] & (1<<i)) {
            int a = aristaIndice[i][0];
            int b = aristaIndice[i][1];
            edgeVert[i] = interp(iso, pos[a], pos[b], val[a], val[b]);
        }
    }

    // Triángulos
    for (int i=0; triTable[cubeIndex][i] != -1; i+=3) {
        int v0 = vertices.size(); vertices.push_back(edgeVert[triTable[cubeIndex][i]]);
        int v1 = vertices.size(); vertices.push_back(edgeVert[triTable[cubeIndex][i+1]]);
        int v2 = vertices.size(); vertices.push_back(edgeVert[triTable[cubeIndex][i+2]]);

        caras.push_back({v0, v2, v1});
    }
}

// -------------------------------------------------------------
// OCTREE — subdivisión adaptativa REAL
// -------------------------------------------------------------
void octree(
    function<double(double,double,double)> f,
    const Vector& minC,
    const Vector& maxC,
    double precision,
    double iso
) {
    double dx = maxC.x - minC.x;
    double dy = maxC.y - minC.y;
    double dz = maxC.z - minC.z;

    // Hoja → procesar cubo
    if (dx <= precision && dy <= precision && dz <= precision) {
        procesarCelda(f, minC, dx, iso);
        return;
    }

    // Subdividir
    Vector mid(
        (minC.x + maxC.x) / 2,
        (minC.y + maxC.y) / 2,
        (minC.z + maxC.z) / 2
    );

    octree(f, minC,               mid,               precision, iso);
    octree(f, Vector(mid.x,minC.y,minC.z), Vector(maxC.x,mid.y,mid.z), precision, iso);
    octree(f, Vector(mid.x,mid.y,minC.z), Vector(maxC.x,maxC.y,mid.z), precision, iso);
    octree(f, Vector(minC.x,mid.y,minC.z), Vector(mid.x,maxC.y,mid.z), precision, iso);

    octree(f, Vector(minC.x,minC.y,mid.z), Vector(mid.x,mid.y,maxC.z), precision, iso);
    octree(f, Vector(mid.x,minC.y,mid.z),  Vector(maxC.x,mid.y,maxC.z), precision, iso);
    octree(f, Vector(mid.x,mid.y,mid.z),   maxC,                         precision, iso);
    octree(f, Vector(minC.x,mid.y,mid.z),  Vector(mid.x,maxC.y,maxC.z), precision, iso);
}

// -----------------------------
// Guardar PLY
// -----------------------------
void guardarPLY(const string& name) {
    ofstream out(name);

    out << "ply\nformat ascii 1.0\n";
    out << "element vertex " << vertices.size() << "\n";
    out << "property float x\nproperty float y\nproperty float z\n";
    out << "element face " << caras.size() << "\n";
    out << "property list uchar int vertex_indices\n";
    out << "end_header\n";

    for (auto& v : vertices)
        out << v.x << " " << v.y << " " << v.z << "\n";

    for (auto& f : caras)
        out << "3 " << f[0] << " " << f[1] << " " << f[2] << "\n";

    cout << "PLY generado: " << name << "\n";
}

// -------------------------------------------------
// Función principal similar a tu draw_curve original
// -------------------------------------------------
void draw_curve_octree(
    function<double(double,double,double)> f,
    string filename,
    double xmin, double ymin, double zmin,
    double xmax, double ymax, double zmax,
    double precision
) {
    vertices.clear();
    caras.clear();

    octree(
        f,
        Vector(xmin,ymin,zmin),
        Vector(xmax,ymax,zmax),
        precision,
        0.0
    );

    guardarPLY(filename);
}

// -----------------------------
// PRUEBA EXACTA A TU ORIGINAL
// -----------------------------
int main() {
    auto f = [](double x,double y,double z){
        double cx=0.5, cy=0.5, cz=0.5, r=0.35;
        return (x-cx)*(x-cx)+(y-cy)*(y-cy)+(z-cz)*(z-cz) - r*r;
    };

    draw_curve_octree(f, "output_octree.ply", 0,0,0, 1,1,1, 0.02);

    return 0;
}
