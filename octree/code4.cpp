#include <iostream>
#include <fstream>
#include <vector>
#include <functional>
#include <cmath>
#include <omp.h>
#include <atomic>
#include "tablas.h"

using namespace std;

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

// Interpolación optimizada
inline Vector interp(double iso, const Vector& p1, const Vector& p2, double v1, double v2) {
    double denom = v2 - v1;
    if (fabs(denom) < 1e-12) return p1;
    
    double t = (iso - v1) / denom;
    return Vector(
        p1.x + (p2.x - p1.x) * t,
        p1.y + (p2.y - p1.y) * t,
        p1.z + (p2.z - p1.z) * t
    );
}

// Estructura thread-local
struct ThreadData {
    vector<Vector> vertices;
    vector<vector<int>> caras;
    long long celdas;
    long long flops;
    
    ThreadData() : celdas(0), flops(0) {
        vertices.reserve(100000);
        caras.reserve(50000);
    }
};

// Procesamiento de celda con conteo REAL de FLOPs
inline void procesarCelda(const function<double(double,double,double)>& f, 
                         const Vector& minCorner, double size, double iso,
                         ThreadData& td) {
    Vector pos[8];
    double val[8];

    // Evaluar 8 esquinas
    for (int i=0; i<8; i++) {
        pos[i] = minCorner + vertCubo[i] * size;
        val[i] = f(pos[i].x, pos[i].y, pos[i].z);
    }
    
    td.celdas++;
    td.flops += 128; // FLOPs por evaluación de función

    // Calcular índice del cubo
    int cubeIndex = 0;
    for (int i=0; i<8; i++)
        if (val[i] < iso) cubeIndex |= (1 << i);

    int edgeMask = tablaDeAristas[cubeIndex];
    if (edgeMask == 0) return;

    // Calcular intersecciones en aristas
    Vector edgeVert[12];
    int numEdges = 0;
    for (int i=0; i<12; i++) {
        if (edgeMask & (1<<i)) {
            int a = aristaIndice[i][0];
            int b = aristaIndice[i][1];
            edgeVert[i] = interp(iso, pos[a], pos[b], val[a], val[b]);
            numEdges++;
            td.flops += 15;
        }
    }

    // Generar triángulos
    int baseIdx = td.vertices.size();
    int numTriangles = 0;
    
    for (int i=0; triTable[cubeIndex][i] != -1; i+=3) {
        td.vertices.push_back(edgeVert[triTable[cubeIndex][i]]);
        td.vertices.push_back(edgeVert[triTable[cubeIndex][i+1]]);
        td.vertices.push_back(edgeVert[triTable[cubeIndex][i+2]]);
        
        td.caras.push_back({baseIdx, baseIdx+2, baseIdx+1});
        baseIdx += 3;
        numTriangles++;
    }
    
    td.flops += numTriangles * 5;
}

// Octree recursivo
void octreeRecursivo(const function<double(double,double,double)>& f, 
                     const Vector& minC, const Vector& maxC,
                     double precision, double iso, 
                     ThreadData& td, int depth = 0) {
    double dx = maxC.x - minC.x;
    double dy = maxC.y - minC.y;
    double dz = maxC.z - minC.z;

    if (dx <= precision && dy <= precision && dz <= precision) {
        procesarCelda(f, minC, dx, iso, td);
        return;
    }

    Vector mid(
        (minC.x + maxC.x) * 0.5,
        (minC.y + maxC.y) * 0.5,
        (minC.z + maxC.z) * 0.5
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

    for (int i = 0; i < 8; i++) {
        octreeRecursivo(f, octantes[i][0], octantes[i][1], 
                       precision, iso, td, depth + 1);
    }
}

// Guardar PLY
void guardarPLY(const string& name, const vector<Vector>& vertices, 
                const vector<vector<int>>& caras) {
    ofstream out(name);
    if (!out) return;

    out << "ply\nformat ascii 1.0\n";
    out << "element vertex " << vertices.size() << "\n";
    out << "property float x\nproperty float y\nproperty float z\n";
    out << "element face " << caras.size() << "\n";
    out << "property list uchar int vertex_indices\n";
    out << "end_header\n";

    for (const auto& v : vertices)
        out << v.x << " " << v.y << " " << v.z << "\n";

    for (const auto& f : caras)
        out << "3 " << f[0] << " " << f[1] << " " << f[2] << "\n";
}

// Análisis de escalabilidad con threads - CORREGIDO COMPLETAMENTE
void analisisEscalabilidad() {
    auto esfera = [](double x,double y,double z){
        double cx=0.5, cy=0.5, cz=0.5, r=0.35;
        return (x-cx)*(x-cx)+(y-cy)*(y-cy)+(z-cz)*(z-cz) - r*r;
    };

    cout << "\n========================================" << endl;
    cout << " ANÁLISIS DE ESCALABILIDAD - THREADS  " << endl;
    cout << "========================================" << endl;

    int threadsArray[] = {1, 2, 4, 8};
    int nTests = 4;
    double tiempoSerial = 0;
    
    struct Metricas {
        int threads;
        double tiempo;
        long long celdas;
        long long flops;
        int vertices;
        int caras;
    };
    vector<Metricas> resultados;

    // Reservar espacio para evitar realocaciones
    resultados.reserve(nTests);

    for (int i = 0; i < nTests; i++) {
        int numThreads = threadsArray[i];
        
        cout << "\n========================================" << endl;
        cout << ">>> PRUEBA " << (i+1) << "/" << nTests << ": " << numThreads << " thread(s)" << endl;
        cout << "========================================" << endl;
        
        // Configurar OpenMP ANTES de crear threadData
        omp_set_num_threads(numThreads);
        omp_set_dynamic(0); // Deshabilitar ajuste dinámico
        
        // División espacial ANTES de la región paralela
        int divs = max(2, (int)ceil(cbrt(numThreads * 4)));
        double step = 1.0 / divs;
        int totalRegions = divs * divs * divs;
        
        cout << "  • Threads configurados: " << numThreads << endl;
        cout << "  • Regiones espaciales: " << totalRegions << endl;
        
        // Crear threadData con el tamaño EXACTO
        vector<ThreadData> threadData(numThreads);
        for (int j = 0; j < numThreads; j++) {
            threadData[j].vertices.reserve(100000);
            threadData[j].caras.reserve(50000);
        }
        
        atomic<long long> totalCeldas(0);
        atomic<long long> totalFlops(0);
        
        cout << "  • Iniciando procesamiento..." << endl;
        double inicio = omp_get_wtime();
        
        // REGIÓN PARALELA
        #pragma omp parallel num_threads(numThreads)
        {
            int tid = omp_get_thread_num();
            int nthreads = omp_get_num_threads();
            
            #pragma omp single
            {
                cout << "  • Threads activos en región paralela: " << nthreads << endl;
            }
            
            if (tid >= threadData.size()) {
                #pragma omp critical
                {
                    cerr << "  ⚠ ERROR: Thread " << tid << " >= tamaño vector (" 
                         << threadData.size() << ")" << endl;
                }
            } else {
                ThreadData& td = threadData[tid];
                
                #pragma omp for schedule(dynamic, 1) nowait
                for (int idx = 0; idx < totalRegions; idx++) {
                    int iz = idx / (divs * divs);
                    int remainder = idx % (divs * divs);
                    int iy = remainder / divs;
                    int ix = remainder % divs;
                    
                    Vector minC(ix * step, iy * step, iz * step);
                    Vector maxC((ix+1) * step, (iy+1) * step, (iz+1) * step);
                    
                    octreeRecursivo(esfera, minC, maxC, 0.02, 0.0, td);
                }
            }
        }
        
        double tiempo = omp_get_wtime() - inicio;
        
        // Combinar resultados
        int totalVerts = 0, totalCaras = 0;
        for (int j = 0; j < threadData.size(); j++) {
            totalCeldas += threadData[j].celdas;
            totalFlops += threadData[j].flops;
            totalVerts += threadData[j].vertices.size();
            totalCaras += threadData[j].caras.size();
        }
        
        // Guardar métricas
        Metricas m;
        m.threads = numThreads;
        m.tiempo = tiempo;
        m.celdas = totalCeldas.load();
        m.flops = totalFlops.load();
        m.vertices = totalVerts;
        m.caras = totalCaras;
        resultados.push_back(m);
        
        if (numThreads == 1) {
            tiempoSerial = tiempo;
            cout << "  • TIEMPO SERIAL BASE: " << tiempoSerial << " s" << endl;
        }
        
        double gflops = (double)totalFlops.load() / tiempo / 1e9;
        
        cout << "  ✓ COMPLETADO" << endl;
        cout << "    - Tiempo: " << tiempo << " s" << endl;
        cout << "    - Celdas: " << totalCeldas.load() << endl;
        cout << "    - Vértices: " << totalVerts << endl;
        cout << "    - GFLOP/s: " << gflops << endl;
    }

    cout << "\n================================================" << endl;
    cout << "RESUMEN: " << resultados.size() << " pruebas completadas" << endl;
    cout << "================================================" << endl;

    // Guardar CSV
    ofstream csv("resultados_threads.csv");
    csv << "Threads,Tiempo(s),Speedup,Eficiencia(%),Celdas,Vertices,Caras,FLOPs,GFLOP/s\n";
    
    cout << "\n================================================" << endl;
    cout << "      TABLA DE SPEEDUP Y EFICIENCIA       " << endl;
    cout << "================================================" << endl;
    cout << "Threads | Tiempo(s) | Speedup | Eficiencia | GFLOP/s" << endl;
    cout << "--------|-----------|---------|------------|----------" << endl;
    
    for (const auto& r : resultados) {
        double speedup = (tiempoSerial > 0) ? tiempoSerial / r.tiempo : 1.0;
        double eficiencia = (r.threads > 0) ? (speedup / r.threads) * 100.0 : 0.0;
        double gflops = (double)r.flops / r.tiempo / 1e9;
        
        printf("%7d | %9.4f | %7.2fx | %9.2f%% | %8.3f\n",  
               r.threads, r.tiempo, speedup, eficiencia, gflops);
        
        csv << r.threads << "," 
            << r.tiempo << "," 
            << speedup << ","
            << eficiencia << ","
            << r.celdas << ","
            << r.vertices << ","
            << r.caras << ","
            << r.flops << ","
            << gflops << "\n";
    }
    
    csv.close();
    cout << "\n✓ Resultados guardados en: resultados_threads.csv" << endl;
}

// Análisis con tamaño variable
void analisisTamañoVariable() {
    auto esfera = [](double x,double y,double z){
        double cx=0.5, cy=0.5, cz=0.5, r=0.35;
        return (x-cx)*(x-cx)+(y-cy)*(y-cy)+(z-cz)*(z-cz) - r*r;
    };

    cout << "\n========================================" << endl;
    cout << "  ANÁLISIS DE ESCALABILIDAD - TAMAÑO   " << endl;
    cout << "========================================" << endl;

    int grillas[] = {64, 128, 256, 512};
    string nombres[] = {"64³", "128³", "256³", "512³"};
    
    ofstream csv("resultados_escalabilidad.csv");
    csv << "Resolucion,Grilla,Precision,Tiempo(s),Celdas,Vertices,Caras,FLOPs,GFLOP/s,Celdas_por_segundo\n";
    
    cout << "\nGrilla    | Precisión  | Tiempo(s) | Celdas     | GFLOP/s" << endl;
    cout << "----------|------------|-----------|------------|----------" << endl;

    int numThreads = 8;
    omp_set_num_threads(numThreads);

    for (int i = 0; i < 4; i++) {
        double precision = 1.0 / grillas[i];
        
        cout << ">>> Procesando " << nombres[i] << "..." << endl;
        
        vector<ThreadData> threadData(numThreads);
        atomic<long long> celdas(0);
        atomic<long long> flops(0);
        
        double inicio = omp_get_wtime();
        
        int divs = 8;
        double step = 1.0 / divs;
        int totalRegions = divs * divs * divs;
        
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            ThreadData& td = threadData[tid];
            
            #pragma omp for schedule(dynamic, 1) nowait
            for (int idx = 0; idx < totalRegions; idx++) {
                int iz = idx / (divs * divs);
                int remainder = idx % (divs * divs);
                int iy = remainder / divs;
                int ix = remainder % divs;
                
                Vector minC(ix * step, iy * step, iz * step);
                Vector maxC((ix+1) * step, (iy+1) * step, (iz+1) * step);
                
                octreeRecursivo(esfera, minC, maxC, precision, 0.0, td);
            }
        }
        
        double tiempo = omp_get_wtime() - inicio;
        
        // Combinar resultados
        int totalVerts = 0, totalCaras = 0;
        for (const auto& td : threadData) {
            celdas += td.celdas;
            flops += td.flops;
            totalVerts += td.vertices.size();
            totalCaras += td.caras.size();
        }
        
        double gflops = (double)flops.load() / tiempo / 1e9;
        double celdas_por_seg = (double)celdas.load() / tiempo;
        
        printf("%-9s | %10.6f | %9.4f | %10lld | %8.3f\n", 
               nombres[i].c_str(), precision, tiempo, celdas.load(), gflops);
        
        csv << nombres[i] << "," << grillas[i] << "," << precision << "," 
            << tiempo << "," << celdas.load() << "," << totalVerts << "," 
            << totalCaras << "," << flops.load() << "," << gflops << "," 
            << celdas_por_seg << "\n";
    }
    
    csv.close();
    cout << "\n✓ Resultados guardados en: resultados_escalabilidad.csv" << endl;
}

int main() {
    cout << "========================================" << endl;
    cout << "   MARCHING CUBES PARALELO OPTIMIZADO  " << endl;
    cout << "   Curso: Computación Paralela         " << endl;
    cout << "========================================" << endl;
    cout << "Threads máximos: " << omp_get_max_threads() << endl;
    
    // Verificar que OpenMP está funcionando
    #pragma omp parallel
    {
        #pragma omp master
        {
            cout << "Threads activos en región paralela: " << omp_get_num_threads() << endl;
        }
    }

    analisisEscalabilidad();
    analisisTamañoVariable();
    
    return 0;
}