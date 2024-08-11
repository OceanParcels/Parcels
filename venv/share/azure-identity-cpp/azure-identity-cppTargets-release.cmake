#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Azure::azure-identity" for configuration "Release"
set_property(TARGET Azure::azure-identity APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(Azure::azure-identity PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libazure-identity.so"
  IMPORTED_SONAME_RELEASE "libazure-identity.so"
  )

list(APPEND _cmake_import_check_targets Azure::azure-identity )
list(APPEND _cmake_import_check_files_for_Azure::azure-identity "${_IMPORT_PREFIX}/lib/libazure-identity.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
