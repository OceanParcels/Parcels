#include <time.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct Particle{
    double depth, lat, lon, dt;
}Particle;

typedef struct UV{
    double u, v;
}UV;


UV fieldset_uv(double time, double depth, double lat, double lon, Particle* p){
    UV res;
    res.u = lon;
    res.v = lat;
    return res;
}


void AdvectionRK4(Particle* particle, double time){
    UV uv1 = fieldset_uv(time, particle->depth, particle->lat, particle->lon, particle);
    double lon1 = particle->lon + uv1.u*.5*particle->dt;
    double lat1 = particle->lat + uv1.v*.5*particle->dt;

    UV uv2 = fieldset_uv(time + .5 * particle->dt, particle->depth, lat1, lon1, particle);
    double lon2 = particle->lon + uv2.u*.5*particle->dt;
    double lat2 = particle->lat + uv2.v*.5*particle->dt;

    UV uv3 = fieldset_uv(time + .5 * particle->dt, particle->depth, lat2, lon2, particle);
    double lon3 = particle->lon + uv3.u*particle->dt;
    double lat3 = particle->lat + uv3.v*particle->dt;

    UV uv4 = fieldset_uv(time + particle->dt, particle->depth, lat3, lon3, particle);
    particle->lon += (uv1.u + 2*uv2.u + 2*uv3.u + uv4.u) / 6. * particle->dt;
    particle->lat += (uv1.v + 2*uv2.v + 2*uv3.v + uv4.v) / 6. * particle->dt;
}

typedef struct PSet{
    Particle* particles;
}PSet;

int main(int argc, char** argv){
    clock_t start, end;
    double time_elapsed;
    PSet pset;
    int n_particle = 100000;
    int n_time = 3000;
    
    pset.particles = (Particle*)malloc(sizeof(Particle)*n_particle);
    
    start = clock();
    for(int ip=0; ip<n_particle; ip++){
        for(int it=0; it<n_time; it++){
            AdvectionRK4(pset.particles+ip, 0);
        }
    }
    end = clock();
    time_elapsed = ((double)(end-start))/CLOCKS_PER_SEC;
    printf("%f\n", time_elapsed);   
}
