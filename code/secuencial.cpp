#include <iostream>
#include <fstream>
#include <vector>
#include <functional>
#include <cmath>
#include "tablas.h"

using namespace std;

struct Vector {
    double x;
    double y;
    double z;
    Vector(double x_=0,double y_=0,double z_=0):x(x_),y(y_),z(z_){}
    Vector operator+(const Vector& v) const { 
        return Vector(x+v.x,y+v.y,z+v.z); 
    }
    Vector operator-(const Vector& v) const { 
        return Vector(x-v.x,y-v.y,z-v.z); 
    }
    Vector operator*(double t) const { 
        return Vector(x*t,y*t,z*t); 
    }
};

Vector interpolacion(double nivelIso, const Vector& p1, const Vector& p2, double valorP1, double valorP2) {
    if (abs(nivelIso-valorP1)<1e-6) {
        return p1;
    }
    if (abs(nivelIso-valorP2)<1e-6) {
        return p2;
    }
    if (abs(valorP1-valorP2)<1e-6) {
        return p1;
    }
    double t=(nivelIso-valorP1)/(valorP2-valorP1);
    return (p1+(p2-p1)*t);
}

vector<Vector> vertices;
vector<vector<int>> caras;

const int aristaIndice[12][2] = { 
    {0,1},{1,2},{2,3},{3,0}, //base inferior
    {4,5},{5,6},{6,7},{7,4}, //base superior
    {0,4},{1,5},{2,6},{3,7}  //aristas verticales
};

const Vector verticesCubo[8] = {
    Vector(0,0,0), Vector(1,0,0), Vector(1,1,0), Vector(0,1,0),
    Vector(0,0,1), Vector(1,0,1), Vector(1,1,1), Vector(0,1,1)
};

extern const int tablaDeAristas[256];
extern const int triTable[256][16];

void marchingCubeCelda(function<double(double,double,double)> funcion, double x, double y, double z, double tamanio, double nivelIso) {
    Vector posicionesCubo[8];
    double valoresCubo[8];
    for(int i=0; i<8; i++) {
        posicionesCubo[i] = Vector(x,y,z)+verticesCubo[i]*tamanio;
        valoresCubo[i] = funcion(posicionesCubo[i].x, posicionesCubo[i].y, posicionesCubo[i].z);
    }
    int indiceCubo = 0;
    for(int i=0; i<8; i++) {
        if(valoresCubo[i]<nivelIso) {
            indiceCubo |= (1<<i);
        }
    }
    if(tablaDeAristas[indiceCubo]==0) {
        return; 
    }
    Vector listaVertices[12];
    for(int i=0; i<12; i++) {
        if(tablaDeAristas[indiceCubo] & (1<<i)) {
            int v1 = aristaIndice[i][0];
            int v2 = aristaIndice[i][1];
            listaVertices[i] = interpolacion(nivelIso, posicionesCubo[v1], posicionesCubo[v2], valoresCubo[v1], valoresCubo[v2]);
        }
    }
    for(int i=0; triTable[indiceCubo][i] != -1; i += 3) {
        int i0=vertices.size(); vertices.push_back(listaVertices[triTable[indiceCubo][i]]);
        int i1=vertices.size(); vertices.push_back(listaVertices[triTable[indiceCubo][i+1]]);
        int i2=vertices.size(); vertices.push_back(listaVertices[triTable[indiceCubo][i+2]]);
        caras.push_back({i0, i2, i1});
    }
}

void subdividir(function<double(double,double,double)> funcion, double xmin, double ymin, double zmin, double xmax, double ymax, double zmax, double precision, double nivelIso) {
    double dx=(xmax-xmin);
    double dy=(ymax-ymin);
    double dz=(zmax-zmin);
    if(dx<=precision && dy<=precision && dz<=precision) {
        marchingCubeCelda(funcion, xmin, ymin, zmin, dx, nivelIso);
    } else {
        double xmed=((xmin+xmax)/2);
        double ymed=((ymin+ymax)/2);
        double zmed=((zmin+zmax)/2);
        //recursivo
        subdividir(funcion,xmin,ymin,zmin,xmed,ymed,zmed,precision, nivelIso);
        subdividir(funcion,xmed,ymin,zmin,xmax,ymed,zmed,precision, nivelIso);
        subdividir(funcion,xmed,ymed,zmin,xmax,ymax,zmed,precision, nivelIso);
        subdividir(funcion,xmin,ymed,zmin,xmed,ymax,zmed,precision, nivelIso);
        subdividir(funcion,xmin,ymin,zmed,xmed,ymed,zmax,precision, nivelIso);
        subdividir(funcion,xmed,ymin,zmed,xmax,ymed,zmax,precision, nivelIso);
        subdividir(funcion,xmed,ymed,zmed,xmax,ymax,zmax,precision, nivelIso);
        subdividir(funcion,xmin,ymed,zmed,xmed,ymax,zmax,precision, nivelIso);
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

    for(auto& v:vertices) {
        out << v.x << " " << v.y << " " << v.z << "\n";
    }
    for(auto& f:caras) {
        out << "3 " << f[0] << " " << f[1] << " " << f[2] << "\n";
    }
    out.close();
}

void draw_curve(function<double(double,double,double)> f, const string& filename, double xmin, double ymin, double zmin, double xmax, double ymax, double zmax, double precision) {
    vertices.clear();
    caras.clear();
    double isolevel = 0.0;
    subdividir(f, xmin, ymin, zmin, xmax, ymax, zmax, precision, isolevel);
    PLY(filename);
    cout << "Archivo PLY: " << filename << " con " << vertices.size() << " vertices y " << caras.size() << " caras." << endl;
}

//Mis pruebas

int main() {
    auto f = [](double x, double y, double z) {
        double cx=0.5, cy=0.5, cz=0.5, r=0.5;
        return (x-cx)*(x-cx)+(y-cy)*(y-cy)+(z-cz)*(z-cz)-r*r;
    };
    draw_curve(f, "output.ply", 0, 0, 0, 1, 1, 1, 0.01);
    return 0;
}
