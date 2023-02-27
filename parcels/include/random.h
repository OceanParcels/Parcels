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
  return (float)rand()/(float)((float)(RAND_MAX) / (high-low)) + low;
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

static inline float parcels_vonmisesvariate(float mu, float kappa)
/* Circular data distribution.                                              */
/* Returns a float between 0 and 2*pi                                       */
/* mu is the mean angle, expressed in radians between 0 and 2*pi, and       */
/* kappa is the concentration parameter, which must be greater than or      */
/* equal to zero.  If kappa is equal to zero, this distribution reduces     */
/* to a uniform random angle over the range 0 to 2*pi.                      */
/* Based upon an algorithm published in: Fisher, N.I.,                      */
/* Statistical Analysis of Circular Data", Cambridge University Press, 1993.*/
{
  float u1, u2, u3, r, s, z, d, f, q, theta;

  if (kappa <= 1e-6){
    return (2.0 * M_PI * (float)rand()/(float)(RAND_MAX));
  }

  s = 0.5 / kappa;
  if (fabs(s) <= FLT_EPSILON * fabs(s)){
    return mu;
  }
  r = s + sqrt(1.0 + s * s);

  do {
    u1 = (float)rand()/(float)(RAND_MAX);
    z = cos(M_PI * u1);

    d = z / (r + z);
    u2 = (float)rand()/(float)(RAND_MAX);
  }  while ( ( u2 >= (1.0 - d * d) ) && ( u2 > (1.0 - d) * exp(d) ) );

  q = 1.0 / r;
  f = (q + z) / (1.0 + q * z);
  u3 = (float)rand()/(float)(RAND_MAX);

  if (u3 > 0.5){
    theta = fmod(mu + acos(f), 2.0*M_PI);
  }
  else {
    theta = fmod(mu - acos(f), 2.0*M_PI);
  }
  if (theta < 0){
    theta = 2.0*M_PI+theta;
  }

  return theta;
}

#ifdef __cplusplus
}
#endif
#endif
