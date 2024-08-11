#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "WebP::sharpyuv" for configuration "Release"
set_property(TARGET WebP::sharpyuv APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(WebP::sharpyuv PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libsharpyuv.so.0.1.0"
  IMPORTED_SONAME_RELEASE "libsharpyuv.so.0"
  )

list(APPEND _cmake_import_check_targets WebP::sharpyuv )
list(APPEND _cmake_import_check_files_for_WebP::sharpyuv "${_IMPORT_PREFIX}/lib/libsharpyuv.so.0.1.0" )

# Import target "WebP::webpdecoder" for configuration "Release"
set_property(TARGET WebP::webpdecoder APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(WebP::webpdecoder PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libwebpdecoder.so.3.1.9"
  IMPORTED_SONAME_RELEASE "libwebpdecoder.so.3"
  )

list(APPEND _cmake_import_check_targets WebP::webpdecoder )
list(APPEND _cmake_import_check_files_for_WebP::webpdecoder "${_IMPORT_PREFIX}/lib/libwebpdecoder.so.3.1.9" )

# Import target "WebP::webp" for configuration "Release"
set_property(TARGET WebP::webp APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(WebP::webp PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libwebp.so.7.1.9"
  IMPORTED_SONAME_RELEASE "libwebp.so.7"
  )

list(APPEND _cmake_import_check_targets WebP::webp )
list(APPEND _cmake_import_check_files_for_WebP::webp "${_IMPORT_PREFIX}/lib/libwebp.so.7.1.9" )

# Import target "WebP::webpdemux" for configuration "Release"
set_property(TARGET WebP::webpdemux APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(WebP::webpdemux PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libwebpdemux.so.2.0.15"
  IMPORTED_SONAME_RELEASE "libwebpdemux.so.2"
  )

list(APPEND _cmake_import_check_targets WebP::webpdemux )
list(APPEND _cmake_import_check_files_for_WebP::webpdemux "${_IMPORT_PREFIX}/lib/libwebpdemux.so.2.0.15" )

# Import target "WebP::libwebpmux" for configuration "Release"
set_property(TARGET WebP::libwebpmux APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(WebP::libwebpmux PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libwebpmux.so.3.1.0"
  IMPORTED_SONAME_RELEASE "libwebpmux.so.3"
  )

list(APPEND _cmake_import_check_targets WebP::libwebpmux )
list(APPEND _cmake_import_check_files_for_WebP::libwebpmux "${_IMPORT_PREFIX}/lib/libwebpmux.so.3.1.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
