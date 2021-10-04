# define cmake minimum requirement

cmake_minimum_required(VERSION 2.8)



# project information 

project(CellTracking)



## Global Libraries ###########################################################################

# Set and define required libraries for all projects


# ITK Libraries

find_package(ITK REQUIRED)

if(ITK_FOUND)
	
include(${ITK_USE_FILE})

endif()


## source files #####################################################################

# set sources and headers files



include_directories(${CMAKE_SOURCE_DIR}/source)

include_directories(${CMAKE_SOURCE_DIR}/header)


set(header 
${CMAKE_SOURCE_DIR}/header/trackingincludes.h
${CMAKE_SOURCE_DIR}/header/trackingfunction.h
${CMAKE_SOURCE_DIR}/header/couple.h
${CMAKE_SOURCE_DIR}/header/correspondancecouple.h
${CMAKE_SOURCE_DIR}/header/tools.h
)

set(source 
${CMAKE_SOURCE_DIR}/source/tracking.cpp 
${CMAKE_SOURCE_DIR}/source/trackingfunction.cpp
${CMAKE_SOURCE_DIR}/source/couple.cpp 
${CMAKE_SOURCE_DIR}/source/correspondancecouple.cpp 
${CMAKE_SOURCE_DIR}/source/tools.cpp 
)


## executable #######################################################################

# define executable and link libraries


add_executable(CellTracking ${header} ${source})

target_link_libraries(CellTracking ${ITK_LIBRARIES})



set_target_properties(CellTracking PROPERTIES
                     	
	RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin/debug/
                     
	RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin/release/
)



## preprocessing compilation ########################################################

# define preprocessing variable



set(Build_ApopLast 1 CACHE BOOL "Processing apopthosis in last" FORCE )
target_compile_definitions(CellTracking PRIVATE APOP_LAST=${Build_ApopLast})
