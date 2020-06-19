cmake_minimum_required(VERSION 3.13)
project(ORB_SLAM2_modules)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "-std=c++11")

IF(CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
ENDIF(CMAKE_COMPILER_IS_GNUCC)

find_package( OpenCV  REQUIRED )
include_directories(
        ${OpenCV_INCLUDE_DIRS}
)
include_directories(${PROJECT_SOURCE_DIR})

add_subdirectory(ComputeKeyPointsOctTree)
add_subdirectory(ORBextractor)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_subdirectory(modules)


#add_executable(test  demo_ForTest.cpp
#        common_include.h )
#target_link_libraries( test ${OpenCV_LIBS} )
#
#
#add_executable(Pyramid  demo_Pyramid.cpp
#        common_include.h )
#target_link_libraries( Pyramid ${OpenCV_LIBS} )
#
#add_executable(ORBextractor demo_ORBextractor.cpp ORBextractor/myORBextractor.cpp
#        common_include.h
#        ORBextractor/myORBextractor.h)
#target_link_libraries( ORBextractor ${OpenCV_LIBS} )
#
#add_executable(ComputeKeyPointsOctTree demo_ComputeKeyPointsOctTree.cpp ComputeKeyPointsOctTree/ComputeKeyPointsOctTree.cpp
#        ComputeKeyPointsOctTree/ComputeKeyPointsOctTree.h
#        ComputeKeyPointsOctTree/DistributeOctTree.cpp)
#target_link_libraries( ComputeKeyPointsOctTree ${OpenCV_LIBS} )