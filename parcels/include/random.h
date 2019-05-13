#ifndef _PARCELS_RANDOM_H
#define _PARCELS_RANDOM_H
#ifdef __cplusplus
extern "C" {
#endif

#ifdef USE_OPENMP
#include <omp.h>
#endif
#include <gsl/gsl_rng.h>

//#define GSL_RNG_TYPE mt19937
//#define GSL_RNG_SEED 0x1UL

/**************************************************/


/**************************************************/
/*   Random number generation (RNG) functions     */
/**************************************************/

extern gsl_rng *prng_state;
#ifdef USE_OPENMP
#pragma omp threadprivate(prng_state)
#endif

static inline void parcels_seed(int seed)
{
  gsl_rng_env_setup();
  const gsl_rng_type *default_rng_type = gsl_rng_default;
  if (prng_state != NULL)
  {
	  gsl_rng_free(prng_state);
	  prng_state = NULL;
  }
  prng_state = gsl_rng_alloc(default_rng_type);
  gsl_rng_set(prng_state, (unsigned long)seed);
}

static inline float parcels_random()
{
  return (float)gsl_rng_uniform(prng_state);
}

static inline float parcels_uniform(float low, float high)
{
  return (float)parcels_random()/(1.0 / (high-low)) + low;
}

static inline int parcels_randint(int low, int high)
{
  return (int)(gsl_rng_get(prng_state) % (high-low)) + low;
}

static inline float parcels_normalvariate(float loc, float scale)
/* Function to create a Gaussian random variable with mean loc and standard deviation scale */
/* Uses Box-Muller transform, adapted from ftp://ftp.taygeta.com/pub/c/boxmuller.c          */
/*     (c) Copyright 1994, Everett F. Carter Jr. Permission is granted by the author to use */
/*     this software for any application provided this copyright notice is preserved.       */
{
  float x1, x2, w, y1;

  do {
    x1 = 2.0 * (float)gsl_rng_uniform(prng_state) - 1.0;
    x2 = 2.0 * (float)gsl_rng_uniform(prng_state) - 1.0;
    w = x1 * x1 + x2 * x2;
  } while ( w >= 1.0 );

  w = sqrt( (-2.0 * log( w ) ) / w );
  y1 = x1 * w;
  return( loc + y1 * scale );
}

static inline float parcels_expovariate(float lamb)
//Function to create an exponentially distributed random variable 
{
  float u;
  //u = (float)rand()/((float)(RAND_MAX) + 1.0);
  u = (float)gsl_rng_uniform(prng_state);
  return (-log(1.0-u)/lamb);
}

/*Function which is called when the library is unloaded from the system.*/
__attribute__((destructor))
static inline void close_library()
{
  if(prng_state != NULL)
    gsl_rng_free(prng_state);
}

#ifdef __cplusplus
}
#endif
#endif
