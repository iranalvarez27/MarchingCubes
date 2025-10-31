#include <iostream>
#include <fstream>
#include <vector>
#include <functional>
#include <cmath>
#include <omp.h> 
#include "tablas.h"
#include <chrono>

using namespace std;

struct Vector {
    double x, y, z;
    Vector(double x_=0,double y_=0,double z_=0):x(x_),y(y_),z(z_){}
    Vector operator+(const Vector& v) const { return Vector(x+v.x,y+v.y,z+v.z); }
    Vector operator-(const Vector& v) const { return Vector(x-v.x,y-v.y,z-v.z); }
    Vector operator*(double t) const { return Vector(x*t,y*t,z*t); }
};

Vector interpolacion(double nivelIso, const Vector& p1, const Vector& p2, double valorP1, double valorP2) {
    if (abs(nivelIso-valorP1)<1e-6) return p1;
    if (abs(nivelIso-valorP2)<1e-6) return p2;
    if (abs(valorP1-valorP2)<1e-6) return p1;
    double t=(nivelIso-valorP1)/(valorP2-valorP1);
    return p1 + (p2-p1)*t;
}
// variables globales
vector<Vector> vertices;
vector<vector<int>> caras;

const int aristaIndice[12][2] = { 
    {0,1},{1,2},{2,3},{3,0},
    {4,5},{5,6},{6,7},{7,4},
    {0,4},{1,5},{2,6},{3,7}
};

const Vector verticesCubo[8] = {
    Vector(0,0,0), Vector(1,0,0), Vector(1,1,0), Vector(0,1,0),
    Vector(0,0,1), Vector(1,0,1), Vector(1,1,1), Vector(0,1,1)
};

extern const int tablaDeAristas[256];
extern const int triTable[256][16];

void marchingCubeCelda(function<double(double,double,double)> funcion, double x, double y, double z, double tamanio, double nivelIso,
                       vector<Vector>& vertices_local, vector<vector<int>>& caras_local) {

    Vector posicionesCubo[8];
    double valoresCubo[8];
    for(int i=0; i<8; i++) {
        posicionesCubo[i] = Vector(x,y,z)+verticesCubo[i]*tamanio;
        valoresCubo[i] = funcion(posicionesCubo[i].x, posicionesCubo[i].y, posicionesCubo[i].z);
    }

    int indiceCubo = 0;
    for(int i=0; i<8; i++)
        if(valoresCubo[i] <= nivelIso)  
            indiceCubo |= (1<<i);

    if(tablaDeAristas[indiceCubo]==0) return;

    Vector listaVertices[12];
    for(int i=0; i<12; i++) {
        if(tablaDeAristas[indiceCubo] & (1<<i)) {
            int v1 = aristaIndice[i][0];
            int v2 = aristaIndice[i][1];
            listaVertices[i] = interpolacion(nivelIso, posicionesCubo[v1], posicionesCubo[v2], valoresCubo[v1], valoresCubo[v2]);
        }
    }

    for(int i=0; triTable[indiceCubo][i] != -1; i += 3) {
        int i0 = vertices_local.size(); 
        vertices_local.push_back(listaVertices[triTable[indiceCubo][i]]);
        int i1 = vertices_local.size(); 
        vertices_local.push_back(listaVertices[triTable[indiceCubo][i+1]]);
        int i2 = vertices_local.size(); 
        vertices_local.push_back(listaVertices[triTable[indiceCubo][i+2]]);
        caras_local.push_back({i0, i2, i1});
    }
}

void subdividir_paralelo(function<double(double,double,double)> funcion,
                         double xmin, double ymin, double zmin,
                         double xmax, double ymax, double zmax,
                         double precision, double nivelIso) {

    int nx = (int)((xmax - xmin) / precision);
    int ny = (int)((ymax - ymin) / precision);
    int nz = (int)((zmax - zmin) / precision);

    // buffers locales por hilo
    vector<vector<Vector>> vertices_threads(omp_get_max_threads());
    vector<vector<vector<int>>> caras_threads(omp_get_max_threads());

    #pragma omp parallel for collapse(3) schedule(static) //static me da mejor tiempo que dynamic
    /*
    La cláusula schedule(dynamic) distribuye las iteraciones entre los hilos
    conforme estos van terminando su trabajo, resultando útil cuando las iteraciones
    tienen una carga de trabajo desigual.
    */
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nz; k++) {
                double x = xmin + i * precision;
                double y = ymin + j * precision;
                double z = zmin + k * precision;
                int tid = omp_get_thread_num();

                marchingCubeCelda(funcion, x, y, z, precision, nivelIso,
                                  vertices_threads[tid], caras_threads[tid]);
            }
        }
    }

    // fusión de resultados de todos los hilos
    for (int t = 0; t < omp_get_max_threads(); t++) {
        int offset = vertices.size();
        vertices.insert(vertices.end(),
                        vertices_threads[t].begin(),
                        vertices_threads[t].end());
        for (auto &f : caras_threads[t]) {
            caras.push_back({f[0] + offset, f[1] + offset, f[2] + offset});
        }
    }
}


void PLY(const string& filename) {
    ofstream out(filename);
    out<<"ply\nformat ascii 1.0\n";
    out<<"element vertex "<<vertices.size()<<"\n";
    out<<"property float x\nproperty float y\nproperty float z\n";
    out<<"element face "<<caras.size()<<"\n";
    out<<"property list uchar int vertex_indices\n";
    out<<"end_header\n";

    for(auto& v:vertices)
        out << v.x << " " << v.y << " " << v.z << "\n";
    for(auto& f:caras)
        out << "3 " << f[0] << " " << f[1] << " " << f[2] << "\n";

    out.close();
}

void draw_curve(function<double(double,double,double)> f, const string& filename,
                double xmin, double ymin, double zmin, double xmax, double ymax, double zmax, double precision) {
    vertices.clear();
    caras.clear();
    double isolevel = 0.0;
    double t_inicio = omp_get_wtime();
    subdividir_paralelo(f, xmin, ymin, zmin, xmax, ymax, zmax, precision, isolevel);
    double t_final = omp_get_wtime();
    cout << "Tiempo de ejecución (segundos): " << t_final - t_inicio << endl;
    PLY(filename);
    cout << "Archivo PLY: " << filename << " con " << vertices.size()
         << " vertices y " << caras.size() << " caras." << endl;
}

int main() {
    auto f = [](double x, double y, double z) {
        double cx=0.5, cy=0.5, cz=0.5, r=0.5;
        return (x-cx)*(x-cx)+(y-cy)*(y-cy)+(z-cz)*(z-cz)-r*r;
    };
    draw_curve(f, "output_huge.ply", -1, -1, -1, 2, 2, 2, 0.01);
    return 0;
}
