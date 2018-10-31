#ifndef H_VORONOI_H
#define H_VORONOI_H

#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>

#include "voronoi_defs.h"
#include "params.h"
#include "knearests.h"

#define cuda_check(x) if (x!=cudaSuccess) {exit(1);}

#define FOR(I,UPPERBND) for(int I = 0; I<int(UPPERBND); ++I)

typedef unsigned char uchar;  // local indices with special values


static const uchar END_OF_LIST = 255;

struct ConvexCell {
    __host__ __device__ ConvexCell(int p_seed, float* p_pts, Status* p_status);
	__host__ __device__ bool is_security_radius_reached(float4 last_neig);
    __host__ __device__ void clip_by_plane(int vid);
    __host__ __device__ void clip_tet_from_points(float4 A, float4 B, float4 C, float4 D);
    __host__ __device__ float4 compute_triangle_point(uchar3 t, bool persp_divide=true) const;
    __host__ __device__ inline  uchar& ith_plane(uchar t, int i);
    __host__ __device__ int new_point(int vid);
    __host__ __device__ void new_triangle(uchar i, uchar j, uchar k);
    __host__ __device__ void compute_boundary();
	__host__ __device__ void get_voro_diagram(ConvexCell& cc, int seed, int* offset, float3* voro_points, char* voro_faces);
    
    __host__ __device__ bool triangle_is_in_conflict(uchar3 t, float4 eqn) const {
//        return triangle_is_in_conflict_double(t, eqn);
        return triangle_is_in_conflict_float(t, eqn);
    }

    __host__ __device__ bool triangle_is_in_conflict_float(uchar3 t, float4 eqn) const;
    __host__ __device__ bool triangle_is_in_conflict_double(uchar3 t, float4 eqn) const;

    
    Status* status;
    uchar nb_t;
    uchar nb_r;
    float* pts;
    int voro_id;
    float4 voro_seed;
    uchar nb_v;
    uchar first_boundary_;
};

struct GlobalStats {

	GlobalStats() { reset(); }

	void reset() {
		nb_clips_before_radius.clear();
		last_clip.clear();
		nb_removed_voro_vertex_per_clip.clear();
		compute_boundary_iter.clear();
		nbv.clear();
		nbt.clear();
		nb_clips_before_radius.resize(1000, 0);
		last_clip.resize(1000, 0);
		nb_removed_voro_vertex_per_clip.resize(1000, 0);
		compute_boundary_iter.resize(1000, 0);
		nbv.resize(1000, 0);
		nbt.resize(1000, 0);
	}

	void start_cell() {
		cur_clip = 6;/* the bbox */
		last_nz_clip = 6;
	}

	void add_clip(int nb_conflict_vertices) {
		cur_clip++;
		if (nb_conflict_vertices != 0) last_nz_clip = cur_clip;
		nb_removed_voro_vertex_per_clip[nb_conflict_vertices]++;
	}

	void add_compute_boundary_iter(int nb_iter) { compute_boundary_iter[nb_iter]++; }

	void end_cell() {
		nb_clips_before_radius[cur_clip]++;
		last_clip[last_nz_clip]++;
	}

	void export_histogram(std::vector<int> h, const std::string& file_name, const std::string& xlabel, const std::string& ylabel) {
		float sum = 0;
		FOR(i, h.size()) sum += .01*float(h[i]);
		if (sum == 0) sum = 1;
		int last = 0;
		FOR(i, h.size() - 1) if (h[i] > 0) last = i;

		char name[1024];
		char cmd[1024];
		static int histid = 0;
#if defined(__linux__)
		sprintf(name, "tmp%d.py", histid++);
		sprintf(cmd, "python3 %s", name);
#else
		sprintf(name, "C:\\DATA\\tmp_%d_.py", histid);
		sprintf(cmd, "python.exe C:\\DATA\\tmp_%d_.py", histid++);
#endif        
		std::ofstream out(name);
		out << "import matplotlib.pyplot as plt\n";
		out << "y = [";
		FOR(i, last) out << float(h[i]) / sum << " , ";
		out << float(h[last]) / sum;
		out << "]\n";
		out << "x = [v-.5 for v in range(len(y))]\n";
		out << "plt.fill_between(x, [0 for i in y], y, step=\"post\", alpha=.8)\n";
		out << "#plt.step(x, y, where='post')\n";
		out << "plt.tick_params(axis='both', which='major', labelsize=12)\n";
		out << "plt.ylabel('" + ylabel + "', fontsize=14)\n";
		out << "plt.xlabel('" + xlabel + "', fontsize=14)\n";
#if defined(__linux__)
		out << "plt.savefig(\"" + file_name + ".pdf\")\n";
#else
		out << "plt.savefig(\"C:/DATA/" + file_name + ".pdf\")\n";
#endif
		out << "#plt.show()\n";
		system(cmd);
	}

	int cur_clip;
	int last_nz_clip;
	std::vector<int> nbv;
	std::vector<int> nbt;
	std::vector<int> compute_boundary_iter;
	std::vector<int> last_clip;
	std::vector<int> nb_clips_before_radius;
	std::vector<int> nb_removed_voro_vertex_per_clip;

	void show() {
		static int num = 0;
		export_histogram(nb_clips_before_radius, std::string("nb_clips_before_radius") + std::to_string(num), "#required clip planes", "proportion %");
		export_histogram(last_clip, std::string("last_usefull_clip") + std::to_string(num), "#last intersecting clip plane", "proportion %");
		export_histogram(nbv, std::string("nbv") + std::to_string(num), "#intersecting clip planes", "proportion %");
		export_histogram(nbt, std::string("nbt") + std::to_string(num), "#Voronoi vertices", "proportion %");
		export_histogram(nb_removed_voro_vertex_per_clip, std::string("nb_removed_voro_vertex_per_clip") + std::to_string(num), "#R", "proportion %");
		export_histogram(compute_boundary_iter, std::string("compute_boundary_iter") + std::to_string(num), "#iter compute void boundary", "proportion %");
		num++;
	}
};

void compute_voro_diagram_GPU(
    std::vector<float>& pts, std::vector<Status> &stat, std::vector<float>& bary, 
	std::vector<float>& tet_pts, std::vector<int>& tet_indices,
    std::vector<int>* KNN = NULL, // Optional: readback K nearest neighbors.
    int nb_Lloyd_iter = 0         // Optional: Lloyd iterations (not implemented ? TO BE CHECKED)
);

void compute_voro_diagram_CPU(
	std::vector<float>& pts, std::vector<Status> &stat, std::vector<float>& bary,
	std::vector<float>& tet_pts, std::vector<int>& tet_indices, std::vector<int>* KNN = NULL,
	int nb_Lloyd_iter = 0
);

#endif // __VORONOI_H__

