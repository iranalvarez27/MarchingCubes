#include <iostream>
#include <fstream>
#include <vector>
#include <functional>
#include <cmath>
#include <omp.h>
#include <algorithm> 
#include "tablas.h" 

using namespace std;

struct Vector {
    double x, y, z;
    Vector(double x_=0, double y_=0, double z_=0) : x(x_), y(y_), z(z_) {}
    Vector operator+(const Vector& v) const { return Vector(x+v.x, y+v.y, z+v.z); }
    Vector operator-(const Vector& v) const { return Vector(x-v.x, y-v.y, z-v.z); }
    Vector operator*(double t) const { return Vector(x*t, y*t, z*t); }
};

const int aristaIndice[12][2] = {
    {0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},{0,4},{1,5},{2,6},{3,7}
};

const Vector vertCubo[8] = {
    Vector(0,0,0), Vector(1,0,0), Vector(1,1,0), Vector(0,1,0),
    Vector(0,0,1), Vector(1,0,1), Vector(1,1,1), Vector(0,1,1)
};

Vector interp(double iso, const Vector& p1, const Vector& p2, double v1, double v2) {
    if (fabs(iso - v1) < 1e-6) return p1;
    if (fabs(iso - v2) < 1e-6) return p2;
    if (fabs(v1 - v2) < 1e-12) return p1;
    double t = (iso - v1) / (v2 - v1);
    return p1 + (p2 - p1) * t;
}

struct ThreadLocalData {
    vector<Vector> vertices;
    vector<vector<int>> caras;
    
    ThreadLocalData() {
        vertices.reserve(20000);  // mayor capacidad
        caras.reserve(20000);
    }
};

vector<Vector> vertices;
vector<vector<int>> caras;

// NUEVO: Estructura para almacenar octantes a procesar
struct OctanteTask {
    Vector minC, maxC;
};

void procesarCelda(function<double(double,double,double)> f, const Vector& minCorner, double size, double iso,
                    ThreadLocalData& localData) {
    Vector pos[8];
    double val[8];
    
    for (int i=0; i<8; i++) {
        pos[i] = minCorner + vertCubo[i] * size;
        val[i] = f(pos[i].x, pos[i].y, pos[i].z);
    }
    
    int cubeIndex = 0;
    for (int i=0; i<8; i++)
        if (val[i] < iso) cubeIndex |= (1 << i);
    
    if (tablaDeAristas[cubeIndex] == 0) return;
    
    Vector edgeVert[12];
    for (int i=0; i<12; i++) {
        if (tablaDeAristas[cubeIndex] & (1<<i)) {
            int a = aristaIndice[i][0];
            int b = aristaIndice[i][1];
            edgeVert[i] = interp(iso, pos[a], pos[b], val[a], val[b]);
        }
    }
    
    for (int i=0; triTable[cubeIndex][i] != -1; i+=3) {
        int v0 = localData.vertices.size(); 
        localData.vertices.push_back(edgeVert[triTable[cubeIndex][i]]);
        int v1 = localData.vertices.size(); 
        localData.vertices.push_back(edgeVert[triTable[cubeIndex][i+1]]);
        int v2 = localData.vertices.size(); 
        localData.vertices.push_back(edgeVert[triTable[cubeIndex][i+2]]);
        localData.caras.push_back({v0, v2, v1});
    }
}

// OPTIMIZACION 1: Version secuencial pura para niveles profundos
void octreeSecuencial(function<double(double,double,double)> f, const Vector& minC, const Vector& maxC,
                      double precision, double iso, ThreadLocalData& localData) {
    double dx = maxC.x - minC.x;
    double dy = maxC.y - minC.y;
    double dz = maxC.z - minC.z;
    
    if (dx <= precision && dy <= precision && dz <= precision) {
        procesarCelda(f, minC, dx, iso, localData);
        return;
    }
    
    Vector mid((minC.x + maxC.x) / 2, (minC.y + maxC.y) / 2, (minC.z + maxC.z) / 2);
    
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
    
    for (int i = 0; i < 8; i++) {
        octreeSecuencial(f, octantes[i][0], octantes[i][1], precision, iso, localData);
    }
}

// OPTIMIZACION 2: Recolectar octantes en nivel especifico
void recolectarOctantes(const Vector& minC, const Vector& maxC, double precision,
                        int nivelActual, int nivelObjetivo, vector<OctanteTask>& octantes) {
    double dx = maxC.x - minC.x;
    
    if (dx <= precision || nivelActual >= nivelObjetivo) {
        octantes.push_back({minC, maxC});
        return;
    }
    
    Vector mid((minC.x + maxC.x) / 2, (minC.y + maxC.y) / 2, (minC.z + maxC.z) / 2);
    
    Vector oct[8][2] = {
        {minC, mid},
        {Vector(mid.x,minC.y,minC.z), Vector(maxC.x,mid.y,mid.z)},
        {Vector(mid.x,mid.y,minC.z), Vector(maxC.x,maxC.y,mid.z)},
        {Vector(minC.x,mid.y,minC.z), Vector(mid.x,maxC.y,mid.z)},
        {Vector(minC.x,minC.y,mid.z), Vector(mid.x,mid.y,maxC.z)},
        {Vector(mid.x,minC.y,mid.z), Vector(maxC.x,mid.y,maxC.z)},
        {Vector(mid.x,mid.y,mid.z), maxC},
        {Vector(minC.x,mid.y,mid.z), Vector(mid.x,maxC.y,maxC.z)}
    };
    
    for (int i = 0; i < 8; i++) {
        recolectarOctantes(oct[i][0], oct[i][1], precision, nivelActual + 1, nivelObjetivo, octantes);
    }
}

// OPTIMIZACION 3: Estrategia - generar work items y procesarlos con for paralelo
double ejecutarMarchingCubes(function<double(double,double,double)> f, double precision, int numThreads) {
    vertices.clear();
    caras.clear();
    
    omp_set_num_threads(numThreads);
    omp_set_dynamic(0);
    
    double tInicio = omp_get_wtime();
    
    // CLAVE: Calcular nivel optimo segun threads y tamaño del problema
   int nivelSubdivision;

    if (precision >= 0.01) {
        nivelSubdivision = 3;       
    }
    else if (precision >= 0.005) {       
        nivelSubdivision = 4;         
    }
    else if (precision >= 0.002) {    
        nivelSubdivision = 4;          
    }
    else {                           
        nivelSubdivision = 3;       
    }

    // Ajustar segun numero de threads
    if (numThreads <= 2) nivelSubdivision = max(2, nivelSubdivision - 1);
    if (numThreads >= 8) nivelSubdivision = min(6, nivelSubdivision + 1);
    
    // Fase 1: Recolectar todos los octantes a procesar
    vector<OctanteTask> octantes;
    recolectarOctantes(Vector(0,0,0), Vector(1,1,1), precision, 0, nivelSubdivision, octantes);
    
    // Fase 2: Procesar octantes en paralelo con schedule dinamico
    vector<ThreadLocalData> threadData(numThreads);
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        
        // CRITICO: schedule(dynamic) con chunk adaptativo
        int chunk = max(1, (int)(octantes.size() / (numThreads * 8)));
        
        #pragma omp for schedule(dynamic, chunk) nowait
        for (size_t i = 0; i < octantes.size(); i++) {
            octreeSecuencial(f, octantes[i].minC, octantes[i].maxC, precision, 0.0, threadData[tid]);
        }
    }
    
    // Fase 3: Merge eficiente de resultados
    for (int tid = 0; tid < numThreads; tid++) {
        int baseIndex = vertices.size();
        vertices.insert(vertices.end(),
                       make_move_iterator(threadData[tid].vertices.begin()),
                       make_move_iterator(threadData[tid].vertices.end()));
        
        for (auto& cara : threadData[tid].caras) {
            caras.push_back({cara[0] + baseIndex, cara[1] + baseIndex, cara[2] + baseIndex});
        }
    }
    
    return omp_get_wtime() - tInicio;
}

void guardarPLY(const string& name, const vector<Vector>& v_data, const vector<vector<int>>& f_data) {
    ofstream out(name);
    out << "ply\nformat ascii 1.0\n";
    out << "element vertex " << v_data.size() << "\n";
    out << "property float x\nproperty float y\nproperty float z\n";
    out << "element face " << f_data.size() << "\n";
    out << "property list uchar int vertex_indices\n";
    out << "end_header\n";

    for (const auto& v : v_data)
        out << v.x << " " << v.y << " " << v.z << "\n";

    for (const auto& f : f_data)
        out << "3 " << f[0] << " " << f[1] << " " << f[2] << "\n";
}

int main() {
    auto esfera = [](double x,double y,double z){
        double cx=0.5, cy=0.5, cz=0.5, r=0.35;
        return (x-cx)*(x-cx)+(y-cy)*(y-cy)+(z-cz)*(z-cz) - r*r;
    };
    
    vector<int> grillas = {64, 96, 128, 192, 256, 512, 640, 768, 896, 1024};
    vector<int> threads_list = {1, 2, 4, 6, 8, 12, 16, 32, 48, 64, 80};
    
    int max_threads = omp_get_max_threads();
    threads_list.erase(
        remove_if(threads_list.begin(), threads_list.end(), 
                  [max_threads](int t) { return t > max_threads; }),
        threads_list.end()
    );
    
    ofstream csv("tiempos_optimizado.csv");
    csv << "Grilla,Threads,Tiempo(s)\n";
    
    for (int grilla : grillas) {
        double precision = 1.0 / grilla;
        
        cout << "Grilla " << grilla << "³:\n";
        
        for (int num_threads : threads_list) {
            double tiempo = ejecutarMarchingCubes(esfera, precision, num_threads);
            
            cout << "  " << num_threads << " threads: " << tiempo << "s\n";
            csv << grilla << "," << num_threads << "," << tiempo << "\n";
        }
        cout << endl;
    }
    
    csv.close();
    guardarPLY("resultado.ply", vertices, caras);
    
    cout << "Tiempos guardados en: tiempos_optimizado.csv\n";
    cout << "Malla guardada en: resultado.ply\n";
    
    return 0;
}
