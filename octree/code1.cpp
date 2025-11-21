#include <iostream>
#include <fstream>
#include <vector>
#include <functional>
#include <cmath>
#include <omp.h>
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
// Procesar celda cúbica — versión con vectores locales
// ----------------------------------------------------------
void procesarCelda(
    function<double(double,double,double)> f,
    const Vector& minCorner,
    double size,
    double iso,
    vector<Vector>& localVertices,
    vector<vector<int>>& localCaras
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

    // Triángulos - guardar en vectores locales
    for (int i=0; triTable[cubeIndex][i] != -1; i+=3) {
        int v0 = localVertices.size(); 
        localVertices.push_back(edgeVert[triTable[cubeIndex][i]]);
        
        int v1 = localVertices.size(); 
        localVertices.push_back(edgeVert[triTable[cubeIndex][i+1]]);
        
        int v2 = localVertices.size(); 
        localVertices.push_back(edgeVert[triTable[cubeIndex][i+2]]);

        localCaras.push_back({v0, v2, v1});
    }
}

// -------------------------------------------------------------
// OCTREE CON TASKS — Recursivo sin mutex
// -------------------------------------------------------------
void octreeTasks(
    function<double(double,double,double)> f,
    const Vector& minC,
    const Vector& maxC,
    double precision,
    double iso,
    vector<Vector>& localVertices,
    vector<vector<int>>& localCaras,
    int depth = 0,
    int maxDepthForTasks = 3
) {
    double dx = maxC.x - minC.x;
    double dy = maxC.y - minC.y;
    double dz = maxC.z - minC.z;

    // Hoja → procesar cubo
    if (dx <= precision && dy <= precision && dz <= precision) {
        procesarCelda(f, minC, dx, iso, localVertices, localCaras);
        return;
    }

    // Subdividir
    Vector mid(
        (minC.x + maxC.x) / 2,
        (minC.y + maxC.y) / 2,
        (minC.z + maxC.z) / 2
    );

    Vector octantes[8][2] = {
        {minC, mid},
        {Vector(mid.x,minC.y,minC.z), Vector(maxC.x,mid.y,mid.z)},
        {Vector(mid.x,mid.y,minC.z), Vector(maxC.x,maxC.y,mid.z)},
        {Vector(minC.x,mid.y,minC.z), Vector(mid.x,maxC.y,mid.z)},
        {Vector(minC.x,minC.y,mid.z), Vector(mid.x,mid.y,maxC.z)},
        {Vector(mid.x,minC.y,mid.z), Vector(maxC.x,mid.y,maxC.z)},
        {Vector(mid.x,mid.y,mid.z), maxC},
        {Vector(minC.x,mid.y,mid.z), Vector(mid.x,maxC.y,maxC.z)}
    };

    // Si estamos en niveles superficiales, crear tasks
    if (depth < maxDepthForTasks) {
        // Vectores para cada octante
        vector<Vector> octVertices[8];
        vector<vector<int>> octCaras[8];

        // Crear tasks para cada octante
        for (int i = 0; i < 8; i++) {
            #pragma omp task firstprivate(i) shared(octVertices, octCaras)
            {
                octreeTasks(f, octantes[i][0], octantes[i][1], precision, iso, 
                           octVertices[i], octCaras[i], depth + 1, maxDepthForTasks);
            }
        }
        
        // Esperar a que todos los tasks terminen
        #pragma omp taskwait

        // Combinar resultados secuencialmente (sin condiciones de carrera)
        for (int i = 0; i < 8; i++) {
            int baseIndex = localVertices.size();
            
            // Agregar vértices
            localVertices.insert(localVertices.end(), 
                                octVertices[i].begin(), 
                                octVertices[i].end());
            
            // Agregar caras ajustando índices
            for (auto& cara : octCaras[i]) {
                localCaras.push_back({
                    cara[0] + baseIndex,
                    cara[1] + baseIndex,
                    cara[2] + baseIndex
                });
            }
        }
    } else {
        // Niveles profundos: sin tasks (secuencial)
        for (int i = 0; i < 8; i++) {
            octreeTasks(f, octantes[i][0], octantes[i][1], precision, iso, 
                       localVertices, localCaras, depth + 1, maxDepthForTasks);
        }
    }
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
// Función principal con tasks (SIN MUTEX)
// -------------------------------------------------
void draw_curve_octree(
    function<double(double,double,double)> f,
    string filename,
    double xmin, double ymin, double zmin,
    double xmax, double ymax, double zmax,
    double precision,
    int numThreads = 8
) {
    vertices.clear();
    caras.clear();

    omp_set_num_threads(numThreads);
    
    cout << "Iniciando Marching Cubes con OpenMP Tasks (sin mutex)..." << endl;
    cout << "Threads: " << numThreads << endl;
    
    double inicio = omp_get_wtime();

    // Región paralela con tasks
    #pragma omp parallel
    {
        #pragma omp single
        {
            cout << "Thread master iniciando..." << endl;
            octreeTasks(f, Vector(xmin,ymin,zmin), Vector(xmax,ymax,zmax), 
                       precision, 0.0, vertices, caras, 0, 3);
        }
    }

    double fin = omp_get_wtime();
    
    cout << "\n=== RESULTADOS ===" << endl;
    cout << "Tiempo: " << (fin - inicio) << " segundos" << endl;
    cout << "Vértices: " << vertices.size() << endl;
    cout << "Caras: " << caras.size() << endl;

    guardarPLY(filename);
}

// -----------------------------
// PRUEBAS
// -----------------------------
int main() {
    cout << "=== MARCHING CUBES - OCTREE CON TASKS ===" << endl;
    cout << "Threads disponibles: " << omp_get_max_threads() << endl << endl;

    // Esfera simple
    auto esfera = [](double x,double y,double z){
        double cx=0.5, cy=0.5, cz=0.5, r=0.35;
        return (x-cx)*(x-cx)+(y-cy)*(y-cy)+(z-cz)*(z-cz) - r*r;
    };

    // Toro
    auto toro = [](double x, double y, double z) {
        double R = 0.3, r = 0.15;
        double cx = 0.5, cy = 0.5, cz = 0.5;
        x -= cx; y -= cy; z -= cz;
        return (R - sqrt(x*x + y*y))*(R - sqrt(x*x + y*y)) + z*z - r*r;
    };

    cout << "--- Test 1: Esfera ---" << endl;
    draw_curve_octree(esfera, "esfera_tasks.ply", 0,0,0, 1,1,1, 0.02, 8);

    cout << "\n--- Test 2: Toro ---" << endl;
    draw_curve_octree(toro, "toro_tasks.ply", 0,0,0, 1,1,1, 0.015, 8);

    return 0;
}