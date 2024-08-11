/* hdf/src/h4config.h.  Generated from h4config.h.in by configure.  */
/* hdf/src/h4config.h.in.  Generated from configure.ac by autoheader.  */

/* Define if building universal (internal helper macro) */
/* #undef H4_AC_APPLE_UNIVERSAL_BUILD */

/* int* greater or equal to 8 bytes */
#define H4_BIG_LONGS 1

/* Define to dummy `main' function (if any) required to link to the Fortran
   libraries. */
/* #undef H4_F77_DUMMY_MAIN */

/* Define to a macro mangling the given C identifier (in lower and upper
   case), which must not contain underscores, for linking with Fortran. */
/* #undef H4_F77_FUNC */

/* As F77_FUNC, but for C identifiers containing underscores. */
/* #undef H4_F77_FUNC_ */

/* Define to 1 if your Fortran compiler doesn't accept -c and -o together. */
/* #undef H4_F77_NO_MINUS_C_MINUS_O */

/* Define if F77 and FC dummy `main' functions are identical. */
/* #undef H4_FC_DUMMY_MAIN_EQ_F77 */

/* Define to 1 if you have the <arpa/inet.h> header file. */
#define H4_HAVE_ARPA_INET_H 1

/* Define to 1 if you have the <arpa/nameser.h> header file. */
#define H4_HAVE_ARPA_NAMESER_H 1

/* Define to 1 if you have the <dlfcn.h> header file. */
#define H4_HAVE_DLFCN_H 1

/* Define to 1 if you have the <fcntl.h> header file. */
#define H4_HAVE_FCNTL_H 1

/* Define to 1 if you have the `fork' function. */
#define H4_HAVE_FORK 1

/* Define to 1 if you have the `htonl' function. */
#define H4_HAVE_HTONL 1

/* Define to 1 if you have the `htons' function. */
#define H4_HAVE_HTONS 1

/* Define to 1 if you have the <inttypes.h> header file. */
#define H4_HAVE_INTTYPES_H 1

/* Define to 1 if you have the <io.h> header file. */
/* #undef H4_HAVE_IO_H */

/* Define to 1 if you have the <jpeglib.h> header file. */
#define H4_HAVE_JPEGLIB_H 1

/* Define to 1 if you have the `jpeg' library (-ljpeg). */
#define H4_HAVE_LIBJPEG 1

/* Define to 1 if you have the `sz' library (-lsz). */
/* #undef H4_HAVE_LIBSZ */

/* Define to 1 if you have the `z' library (-lz). */
#define H4_HAVE_LIBZ 1

/* Define if we support HDF NetCDF APIs version 2.3.2 */
/* #undef H4_HAVE_NETCDF */

/* Define to 1 if you have the <netdb.h> header file. */
#define H4_HAVE_NETDB_H 1

/* Define to 1 if you have the <netinet/in.h> header file. */
#define H4_HAVE_NETINET_IN_H 1

/* Define to 1 if you have the `ntohl' function. */
#define H4_HAVE_NTOHL 1

/* Define to 1 if you have the `ntohs' function. */
#define H4_HAVE_NTOHS 1

/* Define to 1 if you have the <resolv.h> header file. */
#define H4_HAVE_RESOLV_H 1

/* Define to 1 if you have the <stddef.h> header file. */
#define H4_HAVE_STDDEF_H 1

/* Define to 1 if you have the <stdint.h> header file. */
#define H4_HAVE_STDINT_H 1

/* Define to 1 if you have the <stdio.h> header file. */
#define H4_HAVE_STDIO_H 1

/* Define to 1 if you have the <stdlib.h> header file. */
#define H4_HAVE_STDLIB_H 1

/* Define to 1 if you have the <strings.h> header file. */
#define H4_HAVE_STRINGS_H 1

/* Define to 1 if you have the <string.h> header file. */
#define H4_HAVE_STRING_H 1

/* Define to 1 if you have the `system' function. */
#define H4_HAVE_SYSTEM 1

/* Define to 1 if you have the <sys/file.h> header file. */
#define H4_HAVE_SYS_FILE_H 1

/* Define to 1 if you have the <sys/resource.h> header file. */
#define H4_HAVE_SYS_RESOURCE_H 1

/* Define to 1 if you have the <sys/stat.h> header file. */
#define H4_HAVE_SYS_STAT_H 1

/* Define to 1 if you have the <sys/time.h> header file. */
#define H4_HAVE_SYS_TIME_H 1

/* Define to 1 if you have the <sys/types.h> header file. */
#define H4_HAVE_SYS_TYPES_H 1

/* Define if szip has encoder */
/* #undef H4_HAVE_SZIP_ENCODER */

/* Define to 1 if you have the <szlib.h> header file. */
/* #undef H4_HAVE_SZLIB_H */

/* Define to 1 if you have the <unistd.h> header file. */
#define H4_HAVE_UNISTD_H 1

/* Define to 1 if you have the `vfork' function. */
#define H4_HAVE_VFORK 1

/* Define to 1 if you have the `wait' function. */
#define H4_HAVE_WAIT 1

/* Define to 1 if you have the <zlib.h> header file. */
#define H4_HAVE_ZLIB_H 1

/* Define to 1 if you have the `__builtin_bswap16' function. */
/* #undef H4_HAVE___BUILTIN_BSWAP16 */

/* Define to 1 if you have the `__builtin_bswap32' function. */
/* #undef H4_HAVE___BUILTIN_BSWAP32 */

/* Define to 1 if you have the `__builtin_bswap64' function. */
/* #undef H4_HAVE___BUILTIN_BSWAP64 */

/* Define to the sub-directory where libtool stores uninstalled libraries. */
#define H4_LT_OBJDIR ".libs/"

/* Define if deprecated public API symbols are disabled */
/* #undef H4_NO_DEPRECATED_SYMBOLS */

/* Define to 1 if your C compiler doesn't accept -c and -o together. */
/* #undef H4_NO_MINUS_C_MINUS_O */

/* Not using system xdr */
/* #undef H4_NO_SYS_XDR_INC */

/* Name of package */
#define H4_PACKAGE "hdf"

/* Define to the address where bug reports for this package should be sent. */
#define H4_PACKAGE_BUGREPORT "help@hdfgroup.org"

/* Define to the full name of this package. */
#define H4_PACKAGE_NAME "HDF"

/* Define to the full name and version of this package. */
#define H4_PACKAGE_STRING "HDF 4.2.15"

/* Define to the one symbol short name of this package. */
#define H4_PACKAGE_TARNAME "hdf"

/* Define to the home page for this package. */
#define H4_PACKAGE_URL ""

/* Define to the version of this package. */
#define H4_PACKAGE_VERSION "4.2.15"

/* The size of `int*', as computed by sizeof. */
#define H4_SIZEOF_INTP 8

/* Define to 1 if all of the C90 standard headers exist (not just the ones
   required in a freestanding environment). This macro is provided for
   backward compatibility; new code need not use it. */
#define H4_STDC_HEADERS 1

/* Version number of package */
#define H4_VERSION "4.2.15"

/* Define WORDS_BIGENDIAN to 1 if your processor stores words with the most
   significant byte first (like Motorola and SPARC, unlike Intel). */
#if defined AC_APPLE_UNIVERSAL_BUILD
# if defined __BIG_ENDIAN__
#  define WORDS_BIGENDIAN 1
# endif
#else
# ifndef WORDS_BIGENDIAN
/* #  undef WORDS_BIGENDIAN */
# endif
#endif
