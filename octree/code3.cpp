#include <iostream>
#include <fstream>
#include <vector>
#include <functional>
#include <cmath>
#include <omp.h>
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

//interpolacion
Vector interp(double iso, const Vector& p1, const Vector& p2, double v1, double v2) {
    if (fabs(iso - v1) < 1e-6) return p1;
    if (fabs(iso - v2) < 1e-6) return p2;
    if (fabs(v1 - v2) < 1e-12) return p1;

    double t = (iso - v1) / (v2 - v1);
    return p1 + (p2 - p1) * t;
}

// vectores globales
vector<Vector> vertices;
vector<vector<int>> caras;

// metricas de rendimiento
long long totalCeldasProcesadas = 0;
long long totalFLOPs = 0;

// procesar celda cubica con conteo de FLOPs
void procesarCelda(function<double(double,double,double)> f, const Vector& minCorner, double size, double iso,
                    vector<Vector>& localVertices, vector<vector<int>>& localCaras, long long& localCeldas, long long& localFLOPs) {
    Vector pos[8];
    double val[8];

    // esquinas del cubo
    for (int i=0; i<8; i++) {
        pos[i] = minCorner + vertCubo[i] * size;
        val[i] = f(pos[i].x, pos[i].y, pos[i].z);
        localFLOPs += 10; // operaciones por evaluacion de funcion
    }

    localCeldas++;

    // id MC
    int cubeIndex = 0;
    for (int i=0; i<8; i++)
        if (val[i] < iso) cubeIndex |= (1 << i);

    if (tablaDeAristas[cubeIndex] == 0) return;

    // intersecciones
    Vector edgeVert[12];
    for (int i=0; i<12; i++) {
        if (tablaDeAristas[cubeIndex] & (1<<i)) {
            int a = aristaIndice[i][0];
            int b = aristaIndice[i][1];
            edgeVert[i] = interp(iso, pos[a], pos[b], val[a], val[b]);
            localFLOPs += 15; // interpolacion: resta, división, multiplicación
        }
    }

    // triangulos
    for (int i=0; triTable[cubeIndex][i] != -1; i+=3) {
        int v0 = localVertices.size(); 
        localVertices.push_back(edgeVert[triTable[cubeIndex][i]]);
        
        int v1 = localVertices.size(); 
        localVertices.push_back(edgeVert[triTable[cubeIndex][i+1]]);
        
        int v2 = localVertices.size(); 
        localVertices.push_back(edgeVert[triTable[cubeIndex][i+2]]);

        localCaras.push_back({v0, v2, v1});
        localFLOPs += 5; // operaciones de indexado
    }
}

// OCTREE CON TASKS — Recursivo paralelo

void octreeTasks( function<double(double,double,double)> f, const Vector& minC, const Vector& maxC,
                    double precision, double iso, vector<Vector>& localVertices, vector<vector<int>>& localCaras,
                    long long& localCeldas, long long& localFLOPs, int depth = 0, int maxDepthForTasks = 3) {
    double dx = maxC.x - minC.x;
    double dy = maxC.y - minC.y;
    double dz = maxC.z - minC.z;

    // Hoja debe procesar cubo
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

    // tasks para subdivisiones en niveles superficiales
    if (depth < maxDepthForTasks) {
        // vectores locales para cada octante
        vector<Vector> octVertices[8];
        vector<vector<int>> octCaras[8];
        long long octCeldas[8] = {0};
        long long octFLOPs[8] = {0};

        // crear tasks
        for (int i = 0; i < 8; i++) {
            #pragma omp task firstprivate(i) shared(octVertices, octCaras, octCeldas, octFLOPs)
            {
                octreeTasks(f, octantes[i][0], octantes[i][1], precision, iso, 
                           octVertices[i], octCaras[i], octCeldas[i], octFLOPs[i],
                           depth + 1, maxDepthForTasks);
            }
        }
        
        // esperar todos los tasks
        #pragma omp taskwait

        // combinar resultados (región crítica implícita por taskwait)
        for (int i = 0; i < 8; i++) {
            int baseIndex = localVertices.size();
            
            localVertices.insert(localVertices.end(),  octVertices[i].begin(), octVertices[i].end());
            
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
            octreeTasks(f, octantes[i][0], octantes[i][1], precision, iso, localVertices, localCaras, localCeldas, localFLOPs,
                        depth + 1, maxDepthForTasks);
        }
    }
}

//  PLY
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

// Tiempos 

void draw_curve_octree(function<double(double,double,double)> f, string filename, double xmin, double ymin, double zmin,
                        double xmax, double ymax, double zmax, double precision, int numThreads = 8, bool verbose = true) {
    vertices.clear();
    caras.clear();
    totalCeldasProcesadas = 0;
    totalFLOPs = 0;

    omp_set_num_threads(numThreads);
    
    if (verbose) {
        cout << "\n========================================" << endl;
        cout << "MARCHING CUBES - OCTREE PARALELO" << endl;
        cout << "========================================" << endl;
        cout << "Threads: " << numThreads << endl;
        cout << "Precision: " << precision << endl;
        cout << "Dominio: [" << xmin << "," << xmax << "] x [" 
             << ymin << "," << ymax << "] x [" << zmin << "," << zmax << "]" << endl;
    }
    
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
                       precision, 0.0, vertices, caras, celdas, flops, 0, 2);
            
            totalCeldasProcesadas = celdas;
            totalFLOPs = flops;
            
            tComputo = omp_get_wtime() - tComputoInicio;
        }
    }

    double tTotal = omp_get_wtime() - tInicio;
    double tComunicacion = tTotal - tComputo;
    
    if (verbose) {
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
    }

    guardarPLY(filename);
}

// escalabilidad

void analisisEscalabilidad() {
    auto esfera = [](double x,double y,double z){
        double cx=0.5, cy=0.5, cz=0.5, r=0.35;
        return (x-cx)*(x-cx)+(y-cy)*(y-cy)+(z-cz)*(z-cz) - r*r;
    };

    cout << " ANÁLISIS DE ESCALABILIDAD - THREADS  " << endl;

    int threads[] = {1, 2, 4, 8};
    double tiempos[4];
    long long celdas[4];
    long long flops[4];
    int vertices_count[4];
    int caras_count[4];
    double tiempoSerial = 0;

    for (int i = 0; i < 4; i++) {
        cout << "\n>>> Ejecutando con " << threads[i] << " thread(s)..." << endl;
        
        double inicio = omp_get_wtime();
        draw_curve_octree(esfera, "test_" + to_string(threads[i]) + ".ply", 0, 0, 0, 1, 1, 1, 0.02, threads[i]);
        tiempos[i] = omp_get_wtime() - inicio;
        
        celdas[i] = totalCeldasProcesadas;
        flops[i] = totalFLOPs;
        vertices_count[i] = vertices.size();
        caras_count[i] = caras.size();
        
        if (threads[i] == 1) tiempoSerial = tiempos[i];
    }

    // CSV
    ofstream csvThreads("resultados_threads.csv");
    csvThreads << "Threads,Tiempo(s),Speedup,Eficiencia(%),Celdas,Vertices,Caras,FLOPs,GFLOP/s\n";
    
    cout << "      TABLA DE SPEEDUP Y EFICIENCIA       " << endl;
    cout << "Threads | Tiempo(s) | Speedup | Eficiencia" << endl;
    cout << "--------|-----------|---------|------------" << endl;
    
    for (int i = 0; i < 4; i++) {
        double speedup = tiempoSerial / tiempos[i];
        double eficiencia = speedup / threads[i];
        double gflops = (double)flops[i] / tiempos[i] / 1e9;
        
        printf("%7d | %9.4f | %7.2fx | %9.2f%%\n",  threads[i], tiempos[i], speedup, eficiencia * 100);
        
        csvThreads << threads[i] << "," 
                   << tiempos[i] << "," 
                   << speedup << ","
                   << (eficiencia * 100) << ","
                   << celdas[i] << ","
                   << vertices_count[i] << ","
                   << caras_count[i] << ","
                   << flops[i] << ","
                   << gflops << "\n";
    }
    
    csvThreads.close();
    cout << "Resultados guardados en: resultados_threads.csv" << endl;
    cout << endl;
}

// Analisis con tamaño variable (GRILLAS cubicas)

void analisisTamañoVariable() {
    auto esfera = [](double x,double y,double z){
        double cx=0.5, cy=0.5, cz=0.5, r=0.35;
        return (x-cx)*(x-cx)+(y-cy)*(y-cy)+(z-cz)*(z-cz) - r*r;
    };

    cout << "  ANÁLISIS DE ESCALABILIDAD - TAMAÑO     " << endl;

    // Grillas cúbicas: 128^3, 256^3, 384^3, 512^3
    int grillas[] = {128, 256, 384, 512};
    string nombres[] = {"128x128x128", "256x256x256", "384x384x384", "512x512x512"};
    
    // Calcular precision basada en grilla
    // precision = tamaño_dominio / num_celdas
    // Dominio [0,1], entonces precision = 1.0 / grilla
    double precisiones[4];
    for (int i = 0; i < 4; i++) {
        precisiones[i] = 1.0 / grillas[i];
    }

    // Arrays para guardar resultados
    double tiempos[4];
    long long celdas[4];
    long long flops[4];
    int vertices_count[4];
    int caras_count[4];

    cout << "Grilla       |  Precisión | Tiempo(s)  | Celdas    | Vértices" << endl;
    cout << "-------------|------------|------------|-----------|----------" << endl;

    for (int i = 0; i < 4; i++) {
        cout << "\n>>> Procesando grilla " << nombres[i] << " (precision=" << precisiones[i] << ")..." << endl;
        
        double inicio = omp_get_wtime();
        draw_curve_octree(esfera, "malla_" + to_string(grillas[i]) + "cubed.ply", 
                         0, 0, 0, 1, 1, 1, precisiones[i], 8, false);
        tiempos[i] = omp_get_wtime() - inicio;
        
        celdas[i] = totalCeldasProcesadas;
        flops[i] = totalFLOPs;
        vertices_count[i] = vertices.size();
        caras_count[i] = caras.size();
        
        printf("%-12s | %10.6f | %10.4f | %9lld | %8d\n", 
               nombres[i].c_str(),
               precisiones[i],
               tiempos[i],
               celdas[i],
               vertices_count[i]);
    }

    // CSV
    ofstream csvTamaño("resultados_escalabilidad.csv");
    csvTamaño << "Resolucion,Grilla,Precision,Tiempo(s),Celdas,Vertices,Caras,FLOPs,GFLOP/s,Celdas_por_segundo\n";
    
    for (int i = 0; i < 4; i++) {
        double gflops = (double)flops[i] / tiempos[i] / 1e9;
        double celdas_por_seg = (double)celdas[i] / tiempos[i];
        
        csvTamaño << nombres[i] << ","
                  << grillas[i] << ","
                  << precisiones[i] << ","
                  << tiempos[i] << ","
                  << celdas[i] << ","
                  << vertices_count[i] << ","
                  << caras_count[i] << ","
                  << flops[i] << ","
                  << gflops << ","
                  << celdas_por_seg << "\n";
    }
    
    csvTamaño.close();
    cout << "Resultados guardados en: resultados_escalabilidad.csv" << endl;
    cout << endl;
}


// Pruebas completas

int main() {
    cout << "   PROYECTO: MARCHING CUBES PARALELO       " << endl;
    cout << "   Curso: Computación Paralela             " << endl;
    cout << "Threads máximos disponibles: " << omp_get_max_threads() << endl;

    // analisis de escalabilidad con threads
    analisisEscalabilidad();

    // analisis con tamaño variable del problema
    analisisTamañoVariable();    
    return 0;
}