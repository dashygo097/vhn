cmake_minimum_required(VERSION 3.21)
project(hls LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    message(STATUS "Using ccache: ${CCACHE_PROGRAM}")
endif()

if(APPLE)
    set(IS_APPLE TRUE)
    message(STATUS "Building on macOS")
elseif(WIN32)
    set(IS_WINDOWS TRUE)
    message(STATUS "Building on Windows")
else()
    set(IS_UNIX TRUE)
    message(STATUS "Building on Unix-like system")
endif()

# 3rdparty 
include_directories(3rdparty)
add_subdirectory(3rdparty/googletest EXCLUDE_FROM_ALL)

# hls lib
file(GLOB_RECURSE HLS_SOURCES src/*.c src/*.cc src/*.cpp)
file(GLOB_RECURSE HLS_HEADERS include/*.h include/*.hh include/*.hpp)

add_library(hls ${HLS_SOURCES} ${HLS_HEADERS})

target_include_directories(hls PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

set_target_properties(hls PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# Colored printing
function(print_info MSG COLOR)
  if(CMAKE_COLOR_MAKEFILE)
    execute_process(COMMAND printf "\\033[1;${COLOR}m${MSG}\\033[0m\n")
  else()
    message(STATUS "${MSG}")
  endif()
endfunction()

function(make_paths_relative out_var)
    set(result "")
    foreach(path IN LISTS ${ARGV1})
        file(RELATIVE_PATH rel "${CMAKE_SOURCE_DIR}" "${path}")
        list(APPEND result "${rel}")
    endforeach()
    set(${out_var} "${result}" PARENT_SCOPE)
endfunction()

function(make_preview_string LIST_VAR N)
  set(preview "")

  list(LENGTH ${LIST_VAR} list_length)

  foreach(item IN LISTS ${LIST_VAR})
    list(APPEND preview "${item}")
    if("${preview}" MATCHES "^.*;.*;.*$")
      break()
    endif()
  endforeach()

  if(list_length GREATER N)
    math(EXPR remaining_items "${list_length} - ${N}")
    list(APPEND preview "; ...(+${remaining_items} more)")
  endif()

  set(PREVIOUS_SCOPE_VAR "${preview}" PARENT_SCOPE)
endfunction()

# Stastics
print_info("――――――――――――――――――LIB SCOPE――――――――――――――――――" "0")
make_paths_relative(REL_SOURCES HLS_SOURCES)
make_paths_relative(REL_HEADERS HLS_HEADERS)

make_preview_string(REL_HEADERS 3)
print_info("[TRACE] Header files: ${PREVIOUS_SCOPE_VAR}" "94")
make_preview_string(REL_SOURCES 3)
print_info("[TRACE] Source files: ${PREVIOUS_SCOPE_VAR}" "96")

# Testing
include(GoogleTest)
enable_testing()

# Apps
add_subdirectory(tests)
