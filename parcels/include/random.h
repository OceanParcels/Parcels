#ifndef _PARCELS_RANDOM_H
#define _PARCELS_RANDOM_H
#ifdef __cplusplus
extern "C" {
#endif

/**************************************************/


/**************************************************/
/*   Random number generation (RNG) functions     */
/**************************************************/

static inline void parcels_seed(int seed)
{
  srand(seed);
}

static inline float parcels_random()
{
  return (float)rand()/(float)(RAND_MAX);
}

static inline float parcels_uniform(float low, float high)
{
  return (float)rand()/(float)(RAND_MAX / (high-low)) + low;
}

static inline int parcels_randint(int low, int high)
{
  return (rand() % (high-low)) + low;
}

static inline float parcels_normalvariate(float loc, float scale)
/* Function to create a Gaussian random variable with mean loc and standard deviation scale */
/* Uses Box-Muller transform, adapted from ftp://ftp.taygeta.com/pub/c/boxmuller.c          */
/*     (c) Copyright 1994, Everett F. Carter Jr. Permission is granted by the author to use */
/*     this software for any application provided this copyright notice is preserved.       */
{
  float x1, x2, w, y1;

  do {
    x1 = 2.0 * (float)rand()/(float)(RAND_MAX) - 1.0;
    x2 = 2.0 * (float)rand()/(float)(RAND_MAX) - 1.0;
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
  u = (float)rand()/((float)(RAND_MAX) + 1.0);
  return (-log(1.0-u)/lamb);
}

#ifdef __cplusplus
}
#endif
#endif
