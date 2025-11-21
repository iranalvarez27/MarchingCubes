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

// Métricas de rendimiento
long long totalCeldasProcesadas = 0;
long long totalFLOPs = 0;

// ----------------------------------------------------------
// Procesar celda cúbica con conteo de FLOPs
// ----------------------------------------------------------
void procesarCelda(
    function<double(double,double,double)> f,
    const Vector& minCorner,
    double size,
    double iso,
    vector<Vector>& localVertices,
    vector<vector<int>>& localCaras,
    long long& localCeldas,
    long long& localFLOPs
) {
    Vector pos[8];
    double val[8];

    // Esquinas del cubo
    for (int i=0; i<8; i++) {
        pos[i] = minCorner + vertCubo[i] * size;
        val[i] = f(pos[i].x, pos[i].y, pos[i].z);
        localFLOPs += 10; // Operaciones por evaluación de función
    }

    localCeldas++;

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
            localFLOPs += 15; // Interpolación: resta, división, multiplicación
        }
    }

    // Triángulos
    for (int i=0; triTable[cubeIndex][i] != -1; i+=3) {
        int v0 = localVertices.size(); 
        localVertices.push_back(edgeVert[triTable[cubeIndex][i]]);
        
        int v1 = localVertices.size(); 
        localVertices.push_back(edgeVert[triTable[cubeIndex][i+1]]);
        
        int v2 = localVertices.size(); 
        localVertices.push_back(edgeVert[triTable[cubeIndex][i+2]]);

        localCaras.push_back({v0, v2, v1});
        localFLOPs += 5; // Operaciones de indexado
    }
}

// -------------------------------------------------------------
// OCTREE CON TASKS — Recursivo paralelo
// -------------------------------------------------------------
void octreeTasks(
    function<double(double,double,double)> f,
    const Vector& minC,
    const Vector& maxC,
    double precision,
    double iso,
    vector<Vector>& localVertices,
    vector<vector<int>>& localCaras,
    long long& localCeldas,
    long long& localFLOPs,
    int depth = 0,
    int maxDepthForTasks = 3
) {
    double dx = maxC.x - minC.x;
    double dy = maxC.y - minC.y;
    double dz = maxC.z - minC.z;

    // Hoja → procesar cubo
    if (dx <= precision && dy <= precision && dz <= precision) {
        procesarCelda(f, minC, dx, iso, localVertices, localCaras, localCeldas, localFLOPs);
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

    // Crear tasks para subdivisiones en niveles superficiales
    if (depth < maxDepthForTasks) {
        // Vectores locales para cada octante
        vector<Vector> octVertices[8];
        vector<vector<int>> octCaras[8];
        long long octCeldas[8] = {0};
        long long octFLOPs[8] = {0};

        // Crear tasks
        for (int i = 0; i < 8; i++) {
            #pragma omp task firstprivate(i) shared(octVertices, octCaras, octCeldas, octFLOPs)
            {
                octreeTasks(f, octantes[i][0], octantes[i][1], precision, iso, 
                           octVertices[i], octCaras[i], octCeldas[i], octFLOPs[i],
                           depth + 1, maxDepthForTasks);
            }
        }
        
        // Esperar todos los tasks
        #pragma omp taskwait

        // Combinar resultados (región crítica implícita por taskwait)
        for (int i = 0; i < 8; i++) {
            int baseIndex = localVertices.size();
            
            localVertices.insert(localVertices.end(), 
                                octVertices[i].begin(), 
                                octVertices[i].end());
            
            for (auto& cara : octCaras[i]) {
                localCaras.push_back({
                    cara[0] + baseIndex,
                    cara[1] + baseIndex,
                    cara[2] + baseIndex
                });
            }
            
            localCeldas += octCeldas[i];
            localFLOPs += octFLOPs[i];
        }
    } else {
        // Niveles profundos: secuencial
        for (int i = 0; i < 8; i++) {
            octreeTasks(f, octantes[i][0], octantes[i][1], precision, iso, 
                       localVertices, localCaras, localCeldas, localFLOPs,
                       depth + 1, maxDepthForTasks);
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
// Función principal con medición de tiempos
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
    totalCeldasProcesadas = 0;
    totalFLOPs = 0;

    omp_set_num_threads(numThreads);
    
    cout << "\n========================================" << endl;
    cout << "MARCHING CUBES - OCTREE PARALELO" << endl;
    cout << "========================================" << endl;
    cout << "Threads: " << numThreads << endl;
    cout << "Precision: " << precision << endl;
    cout << "Dominio: [" << xmin << "," << xmax << "] x [" 
         << ymin << "," << ymax << "] x [" << zmin << "," << zmax << "]" << endl;
    
    double tInicio = omp_get_wtime();
    double tComputo = 0;

    // Región paralela con tasks
    #pragma omp parallel
    {
        #pragma omp single
        {
            double tComputoInicio = omp_get_wtime();
            
            long long celdas = 0;
            long long flops = 0;
            
            octreeTasks(f, Vector(xmin,ymin,zmin), Vector(xmax,ymax,zmax), 
                       precision, 0.0, vertices, caras, celdas, flops, 0, 3);
            
            totalCeldasProcesadas = celdas;
            totalFLOPs = flops;
            
            tComputo = omp_get_wtime() - tComputoInicio;
        }
    }

    double tTotal = omp_get_wtime() - tInicio;
    double tComunicacion = tTotal - tComputo;
    
    cout << "\n========================================" << endl;
    cout << "RESULTADOS" << endl;
    cout << "========================================" << endl;
    cout << "Tiempo total:        " << tTotal << " s" << endl;
    cout << "Tiempo cómputo:      " << tComputo << " s" << endl;
    cout << "Tiempo comunicación: " << tComunicacion << " s" << endl;
    cout << "----------------------------------------" << endl;
    cout << "Celdas procesadas:   " << totalCeldasProcesadas << endl;
    cout << "Vértices generados:  " << vertices.size() << endl;
    cout << "Caras generadas:     " << caras.size() << endl;
    cout << "----------------------------------------" << endl;
    cout << "FLOPs totales:       " << totalFLOPs << endl;
    cout << "FLOP/s:              " << (double)totalFLOPs / tComputo << endl;
    cout << "GFLOP/s:             " << (double)totalFLOPs / tComputo / 1e9 << endl;
    cout << "========================================\n" << endl;

    guardarPLY(filename);
}

// -----------------------------
// Análisis de escalabilidad
// -----------------------------
void analisisEscalabilidad() {
    auto esfera = [](double x,double y,double z){
        double cx=0.5, cy=0.5, cz=0.5, r=0.35;
        return (x-cx)*(x-cx)+(y-cy)*(y-cy)+(z-cz)*(z-cz) - r*r;
    };

    cout << "\n╔══════════════════════════════════════════╗" << endl;
    cout << "║   ANÁLISIS DE ESCALABILIDAD - THREADS   ║" << endl;
    cout << "╚══════════════════════════════════════════╝\n" << endl;

    int threads[] = {1, 2, 4, 8};
    double tiempos[4];
    double tiempoSerial = 0;

    for (int i = 0; i < 4; i++) {
        cout << "\n>>> Ejecutando con " << threads[i] << " thread(s)..." << endl;
        
        double inicio = omp_get_wtime();
        draw_curve_octree(esfera, "test_" + to_string(threads[i]) + ".ply", 
                         0, 0, 0, 1, 1, 1, 0.02, threads[i]);
        tiempos[i] = omp_get_wtime() - inicio;
        
        if (threads[i] == 1) tiempoSerial = tiempos[i];
    }

    cout << "\n╔══════════════════════════════════════════╗" << endl;
    cout << "║        TABLA DE SPEEDUP Y EFICIENCIA    ║" << endl;
    cout << "╚══════════════════════════════════════════╝" << endl;
    cout << "Threads | Tiempo(s) | Speedup | Eficiencia" << endl;
    cout << "--------|-----------|---------|------------" << endl;
    
    for (int i = 0; i < 4; i++) {
        double speedup = tiempoSerial / tiempos[i];
        double eficiencia = speedup / threads[i];
        printf("%7d | %9.4f | %7.2fx | %9.2f%%\n", 
               threads[i], tiempos[i], speedup, eficiencia * 100);
    }
    cout << endl;
}

// -----------------------------
// Análisis con tamaño variable
// -----------------------------
void analisisTamañoVariable() {
    auto esfera = [](double x,double y,double z){
        double cx=0.5, cy=0.5, cz=0.5, r=0.35;
        return (x-cx)*(x-cx)+(y-cy)*(y-cy)+(z-cz)*(z-cz) - r*r;
    };

    cout << "\n╔══════════════════════════════════════════╗" << endl;
    cout << "║  ANÁLISIS DE ESCALABILIDAD - TAMAÑO     ║" << endl;
    cout << "╚══════════════════════════════════════════╝\n" << endl;

    double precisiones[] = {0.04, 0.02, 0.01, 0.005};
    string nombres[] = {"Gruesa", "Media", "Fina", "Muy Fina"};

    cout << "Resolución |  Tiempo(s)  | Celdas    | Vértices" << endl;
    cout << "-----------|-------------|-----------|----------" << endl;

    for (int i = 0; i < 4; i++) {
        cout << "\n>>> Procesando malla " << nombres[i] << "..." << endl;
        draw_curve_octree(esfera, "malla_" + nombres[i] + ".ply", 
                         0, 0, 0, 1, 1, 1, precisiones[i], 8);
        
        printf("%-10s | %11.4f | %9lld | %8zu\n", 
               nombres[i].c_str(), 
               0.0, // Se puede capturar del output anterior
               totalCeldasProcesadas,
               vertices.size());
    }
    cout << endl;
}

// -----------------------------
// MAIN - Pruebas completas
// -----------------------------
int main() {
    cout << "╔══════════════════════════════════════════╗" << endl;
    cout << "║  PROYECTO: MARCHING CUBES PARALELO      ║" << endl;
    cout << "║  Curso: Computación Paralela            ║" << endl;
    cout << "╚══════════════════════════════════════════╝" << endl;
    cout << "Threads máximos disponibles: " << omp_get_max_threads() << endl;

    // 1. Análisis de escalabilidad con threads
    analisisEscalabilidad();

    // 2. Análisis con tamaño variable del problema
    analisisTamañoVariable();

    cout << "\n✓ Análisis completado." << endl;
    cout << "✓ Revise los archivos .ply generados." << endl;
    
    return 0;
}