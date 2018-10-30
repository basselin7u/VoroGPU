Meshless Voronoi diagram source code. 
To build and run under linux:

mkdir build
cd build/
cmake ../
make
./test_voronoi ../data/1M-blue.xyz 

It produces out.xyz file with barycenters per cell.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

This archive contains auxiliary code:
predicate_generator/ is used to generate the code inside #ifdef USE_ARITHMETIC_FILTER
ComparisonBenchmarks/CGAL/          CGAL test
ComparisonBenchmarks/VORO++/        VORO++ test
ComparisonBenchmarks/geogram/       GEOGRAM test
ComparisonBenchmarks/voroGPUonCPU/  our GPU code modified for an execution a single thread CPU

