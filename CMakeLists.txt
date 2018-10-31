cmake_minimum_required(VERSION 2.8)
find_package(CUDA REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_definitions(-DGEO_STATIC_LIBS)

set(CPU_SOURCES 
    CPU/Delaunay_psm.cpp 
    CPU/convex_cell.cpp
    CPU/voronoi_fallback.cpp    
)

cuda_add_executable(
        test_voronoi voronoi.cu knearests.cu test_voronoi.cu
	${CPU_SOURCES}
        OPTIONS "-arch sm_30 -lineinfo"
)

