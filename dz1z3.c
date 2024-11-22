# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <string.h>
# include <time.h>
# include <omp.h>
# include <math.h>

int main ( int argc, char *argv[] );
double *ccn_compute_points_new ( int n );
double *ccn_compute_points_new_parallel ( int n );
int i4_min ( int i1, int i2 );
double *nc_compute_new ( int n, double x_min, double x_max, double x[] );
double *nc_compute_new_parallel ( int n, double x_min, double x_max, double x[] );
void r8mat_write ( char *output_filename, int m, int n, double table[] );
void rescale ( double a, double b, int n, double x[], double w[] );
void rescale_parallel ( double a, double b, int n, double x[], double w[] );
void rule_write ( int order, char *filename, double x[], double w[], double r[] );
int compare_x_w(double *w,double *x,double *w_par,double *x_par,int n);

int main ( int argc, char *argv[] )
{
  
  double start, end, start_par, end_par;
  double cpu_time_used, cpu_time_used_par;

  double a;
  double b;
  char filename[255];
  int n;
  double *r;
  double *w;
  double *x;
  double *w_par;
  double *x_par;
  double x_max;
  double x_min;


  printf ( "\n" );
  printf ( "CCN_RULE\n" );
  printf ( "  C version\n" );
  printf ( "\n" );
  printf ( "  Compute one of a family of nested Clenshaw Curtis rules\n" );
  printf ( "  for approximating\n" );
  printf ( "    Integral ( -1 <= x <= +1 ) f(x) dx\n" );
  printf ( "  of order N.\n" );
  printf ( "\n" );
  printf ( "  The user specifies N, A, B and FILENAME.\n" );
  printf ( "\n" );
  printf ( "  N is the number of points.\n" );
  printf ( "  A is the left endpoint.\n" );
  printf ( "  B is the right endpoint.\n" );
  printf ( "  FILENAME is used to generate 3 files:\n" );
  printf ( "    filename_w.txt - the weight file\n" );
  printf ( "    filename_x.txt - the abscissa file.\n" );
  printf ( "    filename_r.txt - the region file.\n" );

  if ( 1 < argc )
  {
    n = atoi ( argv[1] );
  }
  else
  {
    printf ( "\n" );
    printf ( "  Enter the value of N (1 or greater)\n" );
    scanf ( "%d", &n );
  }

  if ( 2 < argc )
  {
    a = atof ( argv[2] );
  }
  else
  {
    printf ( "\n" );
    printf ( "  Enter the left endpoint A:\n" );
    scanf ( "%lf", &a );
  }

  if ( 3 < argc )
  {
    b = atof ( argv[3] );
  }
  else
  {
    printf ( "\n" );
    printf ( "  Enter the right endpoint B:\n" );
    scanf ( "%lf", &b );
  }

  if ( 4 < argc )
  {
    strcpy ( filename, argv[4] );
  }
  else
  {
    printf ( "\n" );
    printf ( "  Enter FILENAME, the \"root name\" of the quadrature files.\n" );
    scanf ( "%s", filename );
  }

  printf ( "\n" );
  printf ( "  N = %d\n", n );
  printf ( "  A = %g\n", a );
  printf ( "  B = %g\n", b );
  printf ( "  FILENAME = \"%s\".\n", filename );

  r = ( double * ) malloc ( 2 * sizeof ( double ) );

  r[0] = a;
  r[1] = b;

  // SEQUENTIAL IMPLEMENTATION
  start = omp_get_wtime();

  x = ccn_compute_points_new ( n );

  x_min = -1.0;
  x_max = +1.0;
  w = nc_compute_new ( n, x_min, x_max, x );

  rescale ( a, b, n, x, w );

  end = omp_get_wtime();
  cpu_time_used = end - start;
  printf ( "Time for sequential implementation: %f\n", cpu_time_used );
  //END OF SEQUENTIAL IMPLEMENTATION

  //PARALLEL IMPLEMENTATION
  start_par = omp_get_wtime();

  x_par = ccn_compute_points_new_parallel ( n );

  x_min = -1.0;
  x_max = +1.0;
  w_par = nc_compute_new_parallel ( n, x_min, x_max, x_par );

  rescale_parallel ( a, b, n, x_par, w_par );

  end_par = omp_get_wtime();
  cpu_time_used_par = end_par - start_par;
  printf ( "Time for parallel implementation: %f\n", cpu_time_used_par );
  //END OF PARALLEL IMPLEMENTATION
  
  rule_write ( n, filename, x_par, w_par, r );

  if(compare_x_w(w,x,w_par,x_par,n)) printf("Test PASSED\n");
  else printf("Test FAILED\n");

  printf ( "\n" );
  printf ( "CCN_RULE:\n" );
  printf ( "  Normal end of execution.\n" );
  
  free(r);
  free(x);
  free(w);
  free(x_par);
  free(w_par);

  return 0;
}

double *ccn_compute_points_new_parallel ( int n )
{
  int d;
  int i;
  int k;
  int m;
  double r8_pi = 3.141592653589793;
  int td;
  int tu;
  double *x;

  x = ( double * ) malloc ( n * sizeof ( double ) );

  if ( 1 <= n )
  {
    x[0] = 0.5;
  }

  if ( 2 <= n )
  {
    x[1] = 1.0;
  }

  if ( 3 <= n )
  {
    x[2] = 0.0;
  }

  m = 3;
  d = 2;

  while ( m < n )
  {
    tu = d + 1;
    td = d - 1;

    k = i4_min ( d, n - m );
    
    for ( i = 1; i <= k; i++ )
    {
      if ( ( i % 2 ) == 1 )
      {
        x[m+i-1] = tu / 2.0 / ( double ) ( k );
        tu = tu + 2;
      }
      else
      {
        x[m+i-1] = td / 2.0 / ( double ) ( k );
        td = td - 2;
      }
    }
    m = m + k;
    d = d * 2;
  }
#pragma omp parallel for private(i)
  for ( i = 0; i < n; i++ )
  {
    x[i] = cos ( x[i] * r8_pi );
  }
  x[0] = 0.0;

  if ( 2 <= n )
  {
    x[1] = -1.0;
  }

  if ( 3 <= n )
  {
    x[2] = +1.0;
  }

  return x;
}


int i4_min ( int i1, int i2 )
{
  int value;

  if ( i1 < i2 )
  {
    value = i1;
  }
  else
  {
    value = i2;
  }
  return value;
}


double *nc_compute_new_parallel ( int n, double x_min, double x_max, double x[] )
{
  double *d;
  int i;
  int j;
  int k;
  double *w;
  double yvala;
  double yvalb;

  d = ( double * ) malloc ( n * sizeof ( double ) );
  w = ( double * ) malloc ( n * sizeof ( double ) );
#pragma omp parallel for private(i, j, k, yvala, yvalb)
  for ( i = 0; i < n; i++ )
  {
    for ( j = 0; j < n; j++ )
    {
      d[j] = 0.0;
    }
    d[i] = 1.0;

    for ( j = 2; j <= n; j++ )
    {
      for ( k = j; k <= n; k++ )
      {
        d[n+j-k-1] = ( d[n+j-k-1-1] - d[n+j-k-1] ) / ( x[n+1-k-1] - x[n+j-k-1] );
      }
    }

    for ( j = 1; j <= n - 1; j++ )
    {
      for ( k = 1; k <= n - j; k++ )
      {
        d[n-k-1] = d[n-k-1] - x[n-k-j] * d[n-k];
      }
    }

    yvala = d[n-1] / ( double ) ( n );
    for ( j = n - 2; 0 <= j; j-- )
    {
      yvala = yvala * x_min + d[j] / ( double ) ( j + 1 );
    }
    yvala = yvala * x_min;

    yvalb = d[n-1] / ( double ) ( n );
    for ( j = n - 2; 0 <= j; j-- )
    {
      yvalb = yvalb * x_max + d[j] / ( double ) ( j + 1 );
    }
    yvalb = yvalb * x_max;

    w[i] = yvalb - yvala;
  }

  free ( d );

  return w;
}


void r8mat_write ( char *output_filename, int m, int n, double table[] )
{
  int i;
  int j;
  FILE *output;

  output = fopen ( output_filename, "wt" );

  if ( !output )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "R8MAT_WRITE - Fatal error!\n" );
    fprintf ( stderr, "  Could not open the output file.\n" );
    exit ( 1 );
  }

  for ( j = 0; j < n; j++ )
  {
    for ( i = 0; i < m; i++ )
    {
      fprintf ( output, "  %24.16g", table[i+j*m] );
    }
    fprintf ( output, "\n" );
  }

  fclose ( output );

  return;
}


void rescale_parallel ( double a, double b, int n, double x[], double w[] )
{
  int i;
#pragma omp parallel for private(i)
  for ( i = 0; i < n; i++ )
  {
    x[i] = ( ( a + b ) + ( b - a ) * x[i] ) / 2.0;
    w[i] = ( b - a ) * w[i] / 2.0;
  }
  /*
#pragma omp parallel for private(i)
  for ( i = 0; i < n; i++ )
  {
    w[i] = ( b - a ) * w[i] / 2.0;
  }
  */
  return;
}

void rescale ( double a, double b, int n, double x[], double w[] )
{
  int i;

  for ( i = 0; i < n; i++ )
  {
    x[i] = ( ( a + b ) + ( b - a ) * x[i] ) / 2.0;
  }
  for ( i = 0; i < n; i++ )
  {
    w[i] = ( b - a ) * w[i] / 2.0;
  }
  return;
}

double *ccn_compute_points_new ( int n )
{
  int d;
  int i;
  int k;
  int m;
  double r8_pi = 3.141592653589793;
  int td;
  int tu;
  double *x;

  x = ( double * ) malloc ( n * sizeof ( double ) );

  if ( 1 <= n )
  {
    x[0] = 0.5;
  }

  if ( 2 <= n )
  {
    x[1] = 1.0;
  }

  if ( 3 <= n )
  {
    x[2] = 0.0;
  }

  m = 3;
  d = 2;

  while ( m < n )
  {
    tu = d + 1;
    td = d - 1;

    k = i4_min ( d, n - m );

    for ( i = 1; i <= k; i++ )
    {
      if ( ( i % 2 ) == 1 )
      {
        x[m+i-1] = tu / 2.0 / ( double ) ( k );
        tu = tu + 2;
      }
      else
      {
        x[m+i-1] = td / 2.0 / ( double ) ( k );
        td = td - 2;
      }
    }
    m = m + k;
    d = d * 2;
  }


  for ( i = 0; i < n; i++ )
  {
    x[i] = cos ( x[i] * r8_pi );
  }
  

  x[0] = 0.0;

  if ( 2 <= n )
  {
    x[1] = -1.0;
  }

  if ( 3 <= n )
  {
    x[2] = +1.0;
  }

  return x;
}

double *nc_compute_new ( int n, double x_min, double x_max, double x[] )
{
  double *d;
  int i;
  int j;
  int k;
  double *w;
  double yvala;
  double yvalb;

  d = ( double * ) malloc ( n * sizeof ( double ) );
  w = ( double * ) malloc ( n * sizeof ( double ) );

  for ( i = 0; i < n; i++ )
  {
    for ( j = 0; j < n; j++ )
    {
      d[j] = 0.0;
    }
    d[i] = 1.0;

    for ( j = 2; j <= n; j++ )
    {
      for ( k = j; k <= n; k++ )
      {
        d[n+j-k-1] = ( d[n+j-k-1-1] - d[n+j-k-1] ) / ( x[n+1-k-1] - x[n+j-k-1] );
      }
    }

    for ( j = 1; j <= n - 1; j++ )
    {
      for ( k = 1; k <= n - j; k++ )
      {
        d[n-k-1] = d[n-k-1] - x[n-k-j] * d[n-k];
      }
    }

    yvala = d[n-1] / ( double ) ( n );
    for ( j = n - 2; 0 <= j; j-- )
    {
      yvala = yvala * x_min + d[j] / ( double ) ( j + 1 );
    }
    yvala = yvala * x_min;

    yvalb = d[n-1] / ( double ) ( n );
    for ( j = n - 2; 0 <= j; j-- )
    {
      yvalb = yvalb * x_max + d[j] / ( double ) ( j + 1 );
    }
    yvalb = yvalb * x_max;

    w[i] = yvalb - yvala;
  }

  free ( d );

  return w;
}


void rule_write ( int order, char *filename, double x[], double w[], double r[] )
{
  char filename_r[80];
  char filename_w[80];
  char filename_x[80];

  strcpy ( filename_r, filename );
  strcat ( filename_r, "_r.txt" );
  strcpy ( filename_w, filename );
  strcat ( filename_w, "_w.txt" );
  strcpy ( filename_x, filename );
  strcat ( filename_x, "_x.txt" );

  printf ( "\n" );
  printf ( "  Creating quadrature files.\n" );
  printf ( "\n" );
  printf ( "  Root file name is     \"%s\".\n", filename );
  printf ( "\n" );
  printf ( "  Weight file will be   \"%s\".\n", filename_w );
  printf ( "  Abscissa file will be \"%s\".\n", filename_x );
  printf ( "  Region file will be   \"%s\".\n", filename_r );

  r8mat_write ( filename_w, 1, order, w );
  r8mat_write ( filename_x, 1, order, x );
  r8mat_write ( filename_r, 1, 2,     r );

  return;
}

int compare_x_w(double *w,double *x,double *w_par,double *x_par,int n){
  double epsilon = 0.01;
  for (int i = 0; i < n; ++i) {
      if (fabs(x[i] - x_par[i]) > epsilon || fabs(w[i] - w_par[i]) > epsilon) {
        return 0;
      }
    }
  return 1;
}