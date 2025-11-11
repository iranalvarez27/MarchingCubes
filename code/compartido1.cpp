#include <iostream>
#include <fstream>
#include <vector>
#include <functional>
#include <cmath>
#include <omp.h> 
#include <cstring>
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

// variables globales protegidas más adelante
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
        if(valoresCubo[i] <= nivelIso)  //para cerrar huecos
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

void subdividir_paralelo(function<double(double,double,double)> funcion,double xmin, double ymin, double zmin,double xmax, double ymax, double zmax,double precision, double nivelIso) {
    int nx = (int)((xmax - xmin) / precision);
    int ny = (int)((ymax - ymin) / precision);
    int nz = (int)((zmax - zmin) / precision);

    #pragma omp parallel
    { 
        /*
        se crean los hilos una sola vez. El for se reparte entre esos hilos ya activos. 
        Se evita el costo de copia/join en cada llamada como cuando se usa #pragma omp parallel for porque crea mas copias
        */

        //cada hilo acumula resultados localmente
        vector<Vector> vertices_local;
        vector<vector<int>> caras_local;

        //Bucle paralelo
        #pragma omp for collapse(3) schedule(static) //static me da mejor tiempo que dynamic, pero auto es mejor
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                for (int k = 0; k < nz; k++) {
                    double x = xmin + i * precision;
                    double y = ymin + j * precision;
                    double z = zmin + k * precision;
                    marchingCubeCelda(funcion, x, y, z, precision, nivelIso, vertices_local, caras_local);
                }
            }
        }

        // un solo merge por hilo (no por celda)
        #pragma omp critical
        {
            int offset = vertices.size();
            vertices.insert(vertices.end(), vertices_local.begin(), vertices_local.end());
            for (auto &f : caras_local)
                caras.push_back({f[0]+offset, f[1]+offset, f[2]+offset});
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
    double t_exec = t_final - t_inicio;

    // calcular número total de triángulos generados
    size_t total_triangles = caras.size();

    // salida estándar (para SLURM u otros medidores)
    std::cout << "TIME=" << t_exec << std::endl;
    std::cout << "TRIANGLES=" << total_triangles << std::endl;
    std::cout << std::flush;

    // guardar archivo PLY
    PLY(filename);

    // mensaje informativo
    cout << "Archivo PLY: " << filename
         << " con " << vertices.size() << " vertices y "
         << caras.size() << " caras." << endl;
}


int main(int argc, char** argv) {
    double step = 0.01;
    bool hard = false;

    if (argc > 1) step = atof(argv[1]);
    if (argc > 2 && strcmp(argv[2], "hard") == 0) hard = true;

    function<double(double,double,double)> f;

    if (hard) {
        f = [&](double x,double y,double z) {
            return sin(15*x) + cos(15*y) + sin(15*z) - 0.2;
        };
        cout << "Modo: HARD surface\n";
    } else {
        f = [&](double x,double y,double z) {
            double cx = 0.5, cy = 0.5, cz = 0.5, r = 0.5;
            return (x-cx)*(x-cx)+(y-cy)*(y-cy)+(z-cz)*(z-cz)-r*r;
        };
        cout << "Modo: BASIC sphere\n";
    }

    string file = hard ? "out_hard.ply" : "out_basic.ply";
    draw_curve(f, file, -1, -1, -1, 2, 2, 2, step);
    return 0;
}
