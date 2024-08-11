#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Azure::azure-storage-files-datalake" for configuration "Release"
set_property(TARGET Azure::azure-storage-files-datalake APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(Azure::azure-storage-files-datalake PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libazure-storage-files-datalake.so"
  IMPORTED_SONAME_RELEASE "libazure-storage-files-datalake.so"
  )

list(APPEND _cmake_import_check_targets Azure::azure-storage-files-datalake )
list(APPEND _cmake_import_check_files_for_Azure::azure-storage-files-datalake "${_IMPORT_PREFIX}/lib/libazure-storage-files-datalake.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
