#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>

#include "params.h"
#include "voronoi.h"
#include "stopwatch.h"
#include "CPU/voronoi_fallback.h"

void get_bbox(const std::vector<float>& xyz, float& xmin, float& ymin, float& zmin, float& xmax, float& ymax, float& zmax) {
    int nb_v = xyz.size()/3;
    xmin = xmax = xyz[0];
    ymin = ymax = xyz[1];
    zmin = zmax = xyz[2];
    for(int i=1; i<nb_v; ++i) {
        xmin = std::min(xmin, xyz[3*i]);
        ymin = std::min(ymin, xyz[3*i+1]);
        zmin = std::min(zmin, xyz[3*i+2]);
        xmax = std::max(xmax, xyz[3*i]);
        ymax = std::max(ymax, xyz[3*i+1]);
        zmax = std::max(zmax, xyz[3*i+2]);	    
    }
    float d = xmax-xmin;
    d = std::max(d, ymax-ymin);
    d = std::max(d, zmax-zmin);
    d = 0.001f*d;
    xmin -= d;
    ymin -= d;
    zmin -= d;
    xmax += d;
    ymax += d;
    zmax += d;
}

bool load_file(const char* filename, std::vector<float>& xyz, bool normalize=true) {
    std::ifstream in;
    in.open(filename, std::ifstream::in);
    if (in.fail()) return false;
    std::string line;
    int npts = 0;
    bool firstline = true;
    float x,y,z;
    while (!in.eof()) {
        std::getline(in, line);
        if (!line.length()) continue;
        std::istringstream iss(line.c_str());
        if (firstline) {
            iss >> npts;
            firstline = false;
        } else {
            iss >> x >> y >> z;
            xyz.push_back(x);
            xyz.push_back(y);
            xyz.push_back(z);
        }
    }
    assert(xyz.size() == npts*3);
    in.close();

    if (normalize) { // normalize point cloud between [0,1000]^3
        float xmin,ymin,zmin,xmax,ymax,zmax;
        get_bbox(xyz, xmin, ymin, zmin, xmax, ymax, zmax);

        float maxside = std::max(std::max(xmax-xmin, ymax-ymin), zmax-zmin);
#pragma omp parallel for
        for (int i=0; i<xyz.size()/3; i++) {
            xyz[i*3+0] = 1000.f*(xyz[i*3+0]-xmin)/maxside;
            xyz[i*3+1] = 1000.f*(xyz[i*3+1]-ymin)/maxside;
            xyz[i*3+2] = 1000.f*(xyz[i*3+2]-zmin)/maxside;
        }
        get_bbox(xyz, xmin, ymin, zmin, xmax, ymax, zmax);
        std::cerr << "bbox [" << xmin << ":" << xmax << "], [" << ymin << ":" << ymax << "], [" << zmin << ":" << zmax << "]" << std::endl;
    }
    return true;
}

void drop_xyz_file(std::vector<float>& pts, const char *filename) {
	std::fstream file;
	file.open(filename, std::ios_base::out);
	int k = 0;
	for (unsigned int i = 0; i < pts.size() / 4; i++){
		if (pts[4 * i + 3] > 0) {
			k++;
		}
	}
	file << k << std::endl;
	for (unsigned int i = 0; i < pts.size() / 4; i++){
		if (pts[4 * i + 3] > 0) {
			file << pts[4 * i] / pts[4 * i + 3] << "  " << pts[4 * i + 1] / pts[4 * i + 3] << "  " << pts[4 * i + 2] / pts[4 * i + 3] << std::endl;
		}
	}
    file.close();
}

void printDevProp() {
    
    int devCount; // Number of CUDA devices
    cudaError_t err = cudaGetDeviceCount(&devCount);
    if (err != cudaSuccess) {
        std::cerr << "Failed to initialize CUDA / failed to count CUDA devices (error code << "
		  << cudaGetErrorString(err) << ")! [file: " << __FILE__ << ", line: " <<  __LINE__ << "]" << std::endl;
        exit(1);
    }
    
    printf("CUDA Device Query...\n");
    printf("There are %d CUDA devices.\n", devCount);

    // Iterate through devices
    for (int i=0; i<devCount; ++i) {
        // Get device properties
        printf("\nCUDA Device #%d\n", i);
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        printf("Major revision number:         %d\n",  devProp.major);
        printf("Minor revision number:         %d\n",  devProp.minor);
        printf("Name:                          %s\n",  devProp.name);
        printf("Total global memory:           %lu\n",  devProp.totalGlobalMem);
        printf("Total shared memory per block: %lu\n",  devProp.sharedMemPerBlock);
        printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
        printf("Warp size:                     %d\n",  devProp.warpSize);
        printf("Maximum memory pitch:          %lu\n",  devProp.memPitch);
        printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
        for (int i = 0; i < 3; ++i)
            printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
        for (int i = 0; i < 3; ++i)
            printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
        printf("Clock rate:                    %d\n",  devProp.clockRate);
        printf("Total constant memory:         %lu\n",  devProp.totalConstMem);
        printf("Texture alignment:             %lu\n",  devProp.textureAlignment);
        printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
        printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
        printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
    }
}


int main(int argc, char** argv) {
    initialize_geogram(argc, argv);
    printDevProp();
    if (2>argc) {
        std::cerr << "Usage: " << argv[0] << " points.xyz" << std::endl;
        return 1;
    }
    int *initptr = NULL;
    cudaError_t err = cudaMalloc(&initptr, sizeof(int)); // unused memory, needed for initialize the GPU before time measurements
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate (error code << " << cudaGetErrorString(err) << ")! [file: " << __FILE__ << ", line: " <<  __LINE__ << "]" << std::endl;
        return 1;
    }

    std::vector<float> pts;

    if (!load_file(argv[1], pts, false)) {
        std::cerr << argv[1] << ": could not load file" << std::endl;
        return 1;
    }

    int nb_pts = pts.size()/3;
	std::cout << "number of points" << pts.size()/3 << std::endl;
    
	std::vector<float> tet_pts;
	tet_pts.push_back(100); tet_pts.push_back(100); tet_pts.push_back(100); tet_pts.push_back(0);
	tet_pts.push_back(100); tet_pts.push_back(100); tet_pts.push_back(300); tet_pts.push_back(0);
	tet_pts.push_back(100); tet_pts.push_back(300); tet_pts.push_back(100); tet_pts.push_back(0);
	tet_pts.push_back(100); tet_pts.push_back(300); tet_pts.push_back(300); tet_pts.push_back(0);
	tet_pts.push_back(300); tet_pts.push_back(100); tet_pts.push_back(100); tet_pts.push_back(0);
	tet_pts.push_back(300); tet_pts.push_back(100); tet_pts.push_back(300); tet_pts.push_back(0);
	tet_pts.push_back(300); tet_pts.push_back(300); tet_pts.push_back(100); tet_pts.push_back(0);
	tet_pts.push_back(300); tet_pts.push_back(300); tet_pts.push_back(300); tet_pts.push_back(0);
	tet_pts.push_back(187.95); tet_pts.push_back(230.46); tet_pts.push_back(163.79); tet_pts.push_back(0);

	std::vector<int> tet_indices;
	tet_indices.push_back(0); tet_indices.push_back(2); tet_indices.push_back(3); tet_indices.push_back(8);
	tet_indices.push_back(6); tet_indices.push_back(3); tet_indices.push_back(8); tet_indices.push_back(7);
	tet_indices.push_back(1); tet_indices.push_back(8); tet_indices.push_back(3); tet_indices.push_back(7);
	tet_indices.push_back(1); tet_indices.push_back(8); tet_indices.push_back(5); tet_indices.push_back(0);
	tet_indices.push_back(8); tet_indices.push_back(1); tet_indices.push_back(5); tet_indices.push_back(7);
	tet_indices.push_back(1); tet_indices.push_back(8); tet_indices.push_back(0); tet_indices.push_back(3);
	tet_indices.push_back(5); tet_indices.push_back(8); tet_indices.push_back(4); tet_indices.push_back(0);
	tet_indices.push_back(5); tet_indices.push_back(6); tet_indices.push_back(8); tet_indices.push_back(7);
	tet_indices.push_back(5); tet_indices.push_back(6); tet_indices.push_back(4); tet_indices.push_back(8);
	tet_indices.push_back(4); tet_indices.push_back(2); tet_indices.push_back(0); tet_indices.push_back(8);
	tet_indices.push_back(6); tet_indices.push_back(3); tet_indices.push_back(2); tet_indices.push_back(8);
	tet_indices.push_back(2); tet_indices.push_back(4); tet_indices.push_back(6); tet_indices.push_back(8);
	
	// Choose between GPU and CPU.
	bool gpu = false;
	bool cpu = true;
	if (gpu) {
		std::vector<int> KNN;
		std::vector<float> bary(nb_pts * 4, 0);
		std::vector<Status> stat(nb_pts, security_radius_not_reached);
		compute_voro_diagram_GPU(pts, stat, bary, tet_pts, tet_indices, &KNN);
		// Now computes on the CPU the cells that were not
		// sucessfully computed on the GPU (the ones for
		// which stat[v] != success). 
		//fallback_voro_diagram_CPU(pts, stat, bary, KNN);
		drop_xyz_file(bary, "gpu.xyz");
	}

	if (cpu) {
		std::vector<int> KNN;
		std::vector<float> bary(nb_pts * 4, 0);
		std::vector<Status> stat(nb_pts, security_radius_not_reached);
		compute_voro_diagram_CPU(pts, stat, bary, tet_pts, tet_indices, &KNN);
		//fallback_voro_diagram_CPU(pts, stat, bary, KNN);
		drop_xyz_file(bary, "cpu.xyz");
	}
	
    cudaFree(initptr);
    return 0;
}

