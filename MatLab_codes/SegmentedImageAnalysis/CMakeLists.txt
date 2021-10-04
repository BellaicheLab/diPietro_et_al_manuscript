# define cmake minimum requirement
cmake_minimum_required ( VERSION 2.8 )

# project information 
project( SegmentedImageAnalysis )

## Global Libraries ###########################################################################
# Set and define required libraries for all projects

# ITK Libraries
find_package(ITK REQUIRED)
if(ITK_FOUND)
	include(${ITK_USE_FILE})
endif()

# Boost Libraries
set(Boost_USE_STATIC_LIBS        ON) # only find static libs
set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME    OFF)
find_package(Boost 1.64 COMPONENTS filesystem system program_options REQUIRED)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIR})
    include_directories(${Boost_LIBRARY_DIRS})
endif() 

# Matlab API
find_package(Matlab COMPONENTS MAIN_PROGRAM MAT_LIBRARY MX_LIBRARY REQUIRED)
if(Matlab_FOUND)
	include_directories(${Matlab_INCLUDE_DIRS})
	include_directories(${Matlab_ROOT_DIR})
endif()

find_package(OpenMP)
if(OPENMP_FOUND)
	set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

## source files #####################################################################
# set sources and headers files

include_directories(${CMAKE_SOURCE_DIR}/source)
include_directories(${CMAKE_SOURCE_DIR}/header)

set( source
${CMAKE_SOURCE_DIR}/source/main.cpp 
)

set( header
${CMAKE_SOURCE_DIR}/header/Cell.h
${CMAKE_SOURCE_DIR}/header/Junction.h
${CMAKE_SOURCE_DIR}/header/Utility.h
${CMAKE_SOURCE_DIR}/header/Vertex.h 
${CMAKE_SOURCE_DIR}/header/Matfile.h
)

## executable #######################################################################
# define executable and link libraries

add_executable( SegmentedImageAnalysis ${source} ${header} ${external} )
target_link_libraries( SegmentedImageAnalysis ${ITK_LIBRARIES} ${Boost_LIBRARIES} ${Matlab_LIBRARIES} )

set_target_properties( SegmentedImageAnalysis PROPERTIES
                      RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin/debug/
                      RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin/release/
)
