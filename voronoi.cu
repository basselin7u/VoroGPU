#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>

#include "stopwatch.h"
#include "knearests.h"
#include "voronoi.h"

#ifdef __CUDA_ARCH__
__shared__ uchar3 tr_data[VORO_BLOCK_SIZE * _MAX_T_]; // memory pool for chained lists of triangles
__shared__ uchar boundary_next_data[VORO_BLOCK_SIZE * _MAX_P_];
__shared__ float4 clip_data[VORO_BLOCK_SIZE * _MAX_P_]; // clipping planes

inline  __device__ uchar3& tr(int t) { return  tr_data[threadIdx.x*_MAX_T_ + t]; }
inline  __device__ uchar& boundary_next(int v) { return  boundary_next_data[threadIdx.x*_MAX_P_ + v]; }
inline  __device__ float4& clip(int v) { return  clip_data[threadIdx.x*_MAX_P_ + v]; }
#else
uchar3 tr_data[_MAX_T_];
uchar boundary_next_data[_MAX_P_];
float4 clip_data[_MAX_P_];

inline uchar3& tr(int t) { return  tr_data[t]; }
inline uchar& boundary_next(int v) { return  boundary_next_data[v]; }
inline float4& clip(int v) { return  clip_data[v]; }
GlobalStats gs;
#endif


__host__ __device__ float4 point_from_ptr3(float* f) {
    return make_float4(f[0], f[1], f[2], 1);
}
__host__ __device__ float4 minus4(float4 A, float4 B) {
    return make_float4(A.x-B.x, A.y-B.y, A.z-B.z, A.w-B.w);
}
__host__ __device__ float4 plus4(float4 A, float4 B) {
    return make_float4(A.x+B.x, A.y+B.y, A.z+B.z, A.w+B.w);
}
__host__ __device__ float dot4(float4 A, float4 B) {
    return A.x*B.x + A.y*B.y + A.z*B.z + A.w*B.w;
}
__host__ __device__ float dot3(float4 A, float4 B) {
    return A.x*B.x + A.y*B.y + A.z*B.z;
}
__host__ __device__ float4 mul3(float s, float4 A) {
    return make_float4(s*A.x, s*A.y, s*A.z, 1.);
}
__host__ __device__ float4 cross3(float4 A, float4 B) {
    return make_float4(A.y*B.z - A.z*B.y, A.z*B.x - A.x*B.z, A.x*B.y - A.y*B.x, 0);
}
__host__ __device__ float4 plane_from_point_and_normal(float4 P, float4 n) {
    return  make_float4(n.x, n.y, n.z, -dot3(P, n));
}
__host__ __device__ inline float det2x2(float a11, float a12, float a21, float a22) {
    return a11*a22 - a12*a21;
}
__host__ __device__ inline float det3x3(float a11, float a12, float a13, float a21, float a22, float a23, float a31, float a32, float a33) {
    return a11*det2x2(a22, a23, a32, a33) - a21*det2x2(a12, a13, a32, a33) + a31*det2x2(a12, a13, a22, a23);
}

__host__ __device__ inline float det4x4(
    float a11, float a12, float a13, float a14,
    float a21, float a22, float a23, float a24,               
    float a31, float a32, float a33, float a34,  
    float a41, float a42, float a43, float a44  
) {
    float m12 = a21*a12 - a11*a22;
    float m13 = a31*a12 - a11*a32;
    float m14 = a41*a12 - a11*a42;
    float m23 = a31*a22 - a21*a32;
    float m24 = a41*a22 - a21*a42;
    float m34 = a41*a32 - a31*a42;
    
    float m123 = m23*a13 - m13*a23 + m12*a33;
    float m124 = m24*a13 - m14*a23 + m12*a43;
    float m134 = m34*a13 - m14*a33 + m13*a43;
    float m234 = m34*a23 - m24*a33 + m23*a43;
    
    return (m234*a14 - m134*a24 + m124*a34 - m123*a44);
}   

__host__ __device__ inline double det2x2(double a11, double a12, double a21, double a22) {
    return a11*a22 - a12*a21;
}

__host__ __device__ inline double det3x3(double a11, double a12, double a13, double a21, double a22, double a23, double a31, double a32, double a33) {
    return a11*det2x2(a22, a23, a32, a33) - a21*det2x2(a12, a13, a32, a33) + a31*det2x2(a12, a13, a22, a23);
}

__host__ __device__ inline double det4x4(
    double a11, double a12, double a13, double a14,
    double a21, double a22, double a23, double a24,               
    double a31, double a32, double a33, double a34,  
    double a41, double a42, double a43, double a44  
) {
    double m12 = a21*a12 - a11*a22;
    double m13 = a31*a12 - a11*a32;
    double m14 = a41*a12 - a11*a42;
    double m23 = a31*a22 - a21*a32;
    double m24 = a41*a22 - a21*a42;
    double m34 = a41*a32 - a31*a42;
    
    double m123 = m23*a13 - m13*a23 + m12*a33;
    double m124 = m24*a13 - m14*a23 + m12*a43;
    double m134 = m34*a13 - m14*a33 + m13*a43;
    double m234 = m34*a23 - m24*a33 + m23*a43;
    
    return (m234*a14 - m134*a24 + m124*a34 - m123*a44);
}   

__host__ __device__ inline float get_tet_volume(float4 A, float4 B, float4 C) {
    return -det3x3(A.x, A.y, A.z, B.x, B.y, B.z, C.x, C.y, C.z)/6.;
}
__host__ __device__ void get_tet_volume_and_barycenter(float4& bary, float& volume, float4 A, float4 B, float4 C, float4 D) {
    volume = get_tet_volume(minus4(A, D), minus4(B, D), minus4(C, D));
    bary = make_float4(.25*(A.x+B.x+C.x+D.x), .25*(A.y+B.y+C.y+D.y), .25*(A.z+B.z+C.z+D.z), 1);
}
__host__ __device__ float4 get_plane_from_points(float4 A, float4 B, float4 C) {
	float4 plane = cross3(minus4(B, A), minus4(C, A));
	plane.z = -dot3(plane, A);
	return plane;
}
__host__ __device__ float4 project_on_plane(float4 P, float4 plane) {
    float4 n = make_float4(plane.x, plane.y, plane.z, 0);
    float lambda = (dot4(n, P) + plane.w)/dot4(n, n);
    //    lambda = (dot3(n, P) + plane.w) / norm23(n);
    return plus4(P, mul3(-lambda, n));
}
template <typename T> __host__ __device__ void inline swap(T& a, T& b) { T c(a); a = b; b = c; }


__host__ __device__ ConvexCell::ConvexCell(int p_seed, float* p_pts,Status *p_status) {
    float eps  = .1f;
    float xmin = -eps;
    float ymin = -eps;
    float zmin = -eps;
    float xmax = 1000 + eps;
    float ymax = 1000 + eps;
    float zmax = 1000 + eps;

    pts = p_pts;
    first_boundary_ = END_OF_LIST;
    FOR(i, _MAX_P_) boundary_next(i) = END_OF_LIST;
    voro_id = p_seed;
    voro_seed = make_float4(pts[3 * voro_id], pts[3 * voro_id + 1], pts[3 * voro_id + 2], 1);
    status = p_status;
    *status = success;
	
    clip(0) = make_float4( 1.0,  0.0,  0.0, -xmin);
    clip(1) = make_float4(-1.0,  0.0,  0.0,  xmax);
    clip(2) = make_float4( 0.0,  1.0,  0.0, -ymin);
    clip(3) = make_float4( 0.0, -1.0,  0.0,  ymax);
    clip(4) = make_float4( 0.0,  0.0,  1.0, -zmin);
    clip(5) = make_float4( 0.0,  0.0, -1.0,  zmax);
    nb_v = 6;

    tr(0) = make_uchar3(2, 5, 0);
    tr(1) = make_uchar3(5, 3, 0);
    tr(2) = make_uchar3(1, 5, 2);
    tr(3) = make_uchar3(5, 1, 3);
    tr(4) = make_uchar3(4, 2, 0);
    tr(5) = make_uchar3(4, 0, 3);
    tr(6) = make_uchar3(2, 4, 1);
    tr(7) = make_uchar3(4, 3, 1);
    nb_t = 8;

}

__host__ __device__  bool ConvexCell::is_security_radius_reached(float4 last_neig) {
    // finds furthest voro vertex distance2
    float v_dist = 0;
    FOR(i, nb_t) {
        float4 pc = compute_triangle_point(tr(i));
        float4 diff = minus4(pc, voro_seed);
        float d2 = dot3(diff, diff); // TODO safe to put dot4 here, diff.w = 0
        v_dist = max(d2, v_dist);
    }
    //compare to new neighbors distance2
    float4 diff = minus4(last_neig, voro_seed); // TODO it really should take index of the neighbor instead of the float4, then would be safe to put dot4
    float d2 = dot3(diff, diff);
    return (d2 > 4*v_dist);
}

__host__ __device__ inline  uchar& ConvexCell::ith_plane(uchar t, int i) {
    return reinterpret_cast<uchar *>(&(tr(t)))[i];
}

__host__ __device__ float4 ConvexCell::compute_triangle_point(uchar3 t, bool persp_divide) const {
    float4 pi1 = clip(t.x);
    float4 pi2 = clip(t.y);
    float4 pi3 = clip(t.z);
    float4 result;
    result.x = -det3x3(pi1.w, pi1.y, pi1.z, pi2.w, pi2.y, pi2.z, pi3.w, pi3.y, pi3.z);
    result.y = -det3x3(pi1.x, pi1.w, pi1.z, pi2.x, pi2.w, pi2.z, pi3.x, pi3.w, pi3.z);
    result.z = -det3x3(pi1.x, pi1.y, pi1.w, pi2.x, pi2.y, pi2.w, pi3.x, pi3.y, pi3.w);
    result.w =  det3x3(pi1.x, pi1.y, pi1.z, pi2.x, pi2.y, pi2.z, pi3.x, pi3.y, pi3.z);
    if (persp_divide) return make_float4(result.x / result.w, result.y / result.w, result.z / result.w, 1);
    return result;
}

inline __host__ __device__ float max4(float a, float b, float c, float d) {
    return fmaxf(fmaxf(a,b),fmaxf(c,d));
}

inline __host__ __device__ void get_minmax3(
    float& m, float& M, float x1, float x2, float x3
) {
    m = fminf(fminf(x1,x2), x3);
    M = fmaxf(fmaxf(x1,x2), x3);
}

inline __host__ __device__ double max4(double a, double b, double c, double d) {
    return fmax(fmax(a,b),fmax(c,d));
}

inline __host__ __device__ void get_minmax3(
    double& m, double& M, double x1, double x2, double x3
) {
    m = fmin(fmin(x1,x2), x3);
    M = fmax(fmax(x1,x2), x3);
}


__host__ __device__ bool ConvexCell::triangle_is_in_conflict_float(uchar3 t, float4 eqn) const {
    float4 pi1 = clip(t.x);
    float4 pi2 = clip(t.y);
    float4 pi3 = clip(t.z);
    float det = det4x4(
		pi1.x, pi2.x, pi3.x, eqn.x,
		pi1.y, pi2.y, pi3.y, eqn.y,
		pi1.z, pi2.z, pi3.z, eqn.z,
		pi1.w, pi2.w, pi3.w, eqn.w
    );

#ifdef USE_ARITHMETIC_FILTER
    float maxx = max4(fabsf(pi1.x), fabsf(pi2.x), fabsf(pi3.x), fabsf(eqn.x));
    float maxy = max4(fabsf(pi1.y), fabsf(pi2.y), fabsf(pi3.y), fabsf(eqn.y));    
    float maxz = max4(fabsf(pi1.z), fabsf(pi2.z), fabsf(pi3.z), fabsf(eqn.z));    

    // The constant is computed by the program 
    // in predicate_generator/
    float eps = 6.6876506e-05 * maxx * maxy * maxz;
    
    float min_max;
    float max_max;
    get_minmax3(min_max, max_max, maxx, maxy, maxz);

    eps *= (max_max * max_max);

    if(fabsf(det) < eps) {
	*status = needs_exact_predicates;
    }
#endif

    return (det > 0.0f);
}

__host__ __device__ bool ConvexCell::triangle_is_in_conflict_double(uchar3 t, float4 eqn_f) const {
    float4 pi1_f = clip(t.x);
    float4 pi2_f = clip(t.y);
    float4 pi3_f = clip(t.z);

    double4 eqn = make_double4(eqn_f.x, eqn_f.y, eqn_f.z, eqn_f.w);
    double4 pi1 = make_double4(pi1_f.x, pi1_f.y, pi1_f.z, pi1_f.w);
    double4 pi2 = make_double4(pi2_f.x, pi2_f.y, pi2_f.z, pi2_f.w);
    double4 pi3 = make_double4(pi3_f.x, pi3_f.y, pi3_f.z, pi3_f.w);        
    
    double det = det4x4(
	pi1.x, pi2.x, pi3.x, eqn.x,
	pi1.y, pi2.y, pi3.y, eqn.y,
	pi1.z, pi2.z, pi3.z, eqn.z,
	pi1.w, pi2.w, pi3.w, eqn.w
    );

#ifdef USE_ARITHMETIC_FILTER
    double maxx = max4(fabs(pi1.x), fabs(pi2.x), fabs(pi3.x), fabs(eqn.x));
    double maxy = max4(fabs(pi1.y), fabs(pi2.y), fabs(pi3.y), fabs(eqn.y));    
    double maxz = max4(fabs(pi1.z), fabs(pi2.z), fabs(pi3.z), fabs(eqn.z));    

    // The constant is computed by the program 
    // in predicate_generator/
    double eps = 1.2466136531027298e-13 * maxx * maxy * maxz;
    
    double min_max;
    double max_max;
    get_minmax3(min_max, max_max, maxx, maxy, maxz);

    eps *= (max_max * max_max);

    if(fabs(det) < eps) {
	*status = needs_exact_predicates;
    }
#endif    
    
    return (det > 0.0f);
}

__host__ __device__ void ConvexCell::new_triangle(uchar i, uchar j, uchar k) {
    if (nb_t+1 >= _MAX_T_) { 
        *status = triangle_overflow; 
        return; 
    }
    tr(nb_t) = make_uchar3(i, j, k);
    nb_t++;
}

__host__ __device__ int ConvexCell::new_point(int vid) {
    if (nb_v >= _MAX_P_) { 
        *status = vertex_overflow; 
        return -1; 
    }

    float4 B = point_from_ptr3(pts + 3 * vid);
    float4 dir = minus4(voro_seed, B);
    float4 ave2 = plus4(voro_seed, B);
    float dot = dot3(ave2,dir); // TODO safe to put dot4 here, dir.w = 0
    clip(nb_v) = make_float4(dir.x, dir.y, dir.z, -dot / 2.f);
    nb_v++;
    return nb_v - 1;
}

__host__ __device__ void ConvexCell::compute_boundary() {
    // clean circular list of the boundary
    FOR(i, _MAX_P_) boundary_next(i) = END_OF_LIST;
    first_boundary_ = END_OF_LIST;

    int nb_iter = 0;
    uchar t = nb_t;
#ifndef __CUDA_ARCH__
	if (nb_r>20)
	std::cerr << "nb_t " << (int)nb_t << " nb_r " << (int)nb_r << std::endl;
#endif
    while (nb_r>0) {
        if (nb_iter++>100) { 
            *status = inconsistent_boundary; 
            return; 
        }
        bool is_in_border[3];
        bool next_is_opp[3];
        FOR(e, 3)   is_in_border[e] = (boundary_next(ith_plane(t, e)) != END_OF_LIST);
        FOR(e, 3)   next_is_opp[e] = (boundary_next(ith_plane(t, (e + 1) % 3)) == ith_plane(t, e));

        bool new_border_is_simple = true;
        // check for non manifoldness
        FOR(e, 3) if (!next_is_opp[e] && !next_is_opp[(e + 1) % 3] && is_in_border[(e + 1) % 3]) new_border_is_simple = false;

        // check for more than one boundary ... or first triangle
        if (!next_is_opp[0] && !next_is_opp[1] && !next_is_opp[2]) {
            if (first_boundary_ == END_OF_LIST) {
                FOR(e, 3) boundary_next(ith_plane(t, e)) = ith_plane(t, (e + 1) % 3);
                first_boundary_ = tr(t).x;
            }
            else new_border_is_simple = false;
        }

        if (!new_border_is_simple) {
            t++;
            if (t == nb_t + nb_r) t = nb_t;
            continue;
        }

        // link next
        FOR(e, 3) if (!next_is_opp[e]) boundary_next(ith_plane(t, e)) = ith_plane(t, (e + 1) % 3);

        // destroy link from removed vertices
        FOR(e, 3)  if (next_is_opp[e] && next_is_opp[(e + 1) % 3]) {
            if (first_boundary_ == ith_plane(t, (e + 1) % 3)) first_boundary_ = boundary_next(ith_plane(t, (e + 1) % 3));
            boundary_next(ith_plane(t, (e + 1) % 3)) = END_OF_LIST;
        }

        //remove triangle from R, and restart iterating on R
        swap(tr(t), tr(nb_t+nb_r-1));
        t = nb_t;
        nb_r--;
    }

	IF_CPU(gs.add_compute_boundary_iter(nb_iter);)
}

__host__ __device__ void  ConvexCell::clip_by_plane(int cur_v) {
    if (*status == vertex_overflow) return;
    float4 eqn = clip(cur_v);
    nb_r = 0;

    int i = 0;
    while (i < nb_t) { // for all vertices of the cell
	if(triangle_is_in_conflict(tr(i), eqn)) {
            nb_t--;
            swap(tr(i), tr(nb_t));
            nb_r++;
        }
        else i++;
    }
	if (nb_t < 1) {
		*status = empty_cell;
		return;
	}

	IF_CPU(gs.add_clip(nb_r);)

    if (*status == needs_exact_predicates) {
	return;
    }
    
    if (nb_r == 0) { // if no clips, then remove the plane equation
        nb_v--;
        return;
    }

    // Step 2: compute cavity boundary
    compute_boundary();
    if (*status != success) return;
    if (first_boundary_ == END_OF_LIST) return;

    // Step 3: Triangulate cavity
    uchar cir = first_boundary_;
    do {
        new_triangle(cur_v, cir, boundary_next(cir));
#ifndef __CUDA_ARCH__
		if (nb_t >= _MAX_T_) {
			std::cerr << "erreur grave" << std::endl;
		}
		if (cur_v >= _MAX_P_ || cir >= _MAX_P_ || boundary_next(cir) >= _MAX_P_ ) {
			std::cerr << "erreur grave, triangle: " << (int)cur_v << " " << (int)cir << " " << (int)boundary_next(cir) << std::endl;
			break;
		}
#endif
        if (*status != success) return;
        cir = boundary_next(cir);
    } while (cir != first_boundary_);
}

__host__ __device__ void get_tet_decomposition_of_vertex(ConvexCell& cc, int t, float4* P) {
    float4 C = cc.voro_seed;
    float4 A = cc.compute_triangle_point(tr(t));
    FOR(i,3)  P[2*i  ] = project_on_plane(C, clip(cc.ith_plane(t,i)));
    FOR(i, 3) P[2*i+1] = project_on_plane(A, plane_from_point_and_normal(C, cross3(minus4(P[2*i], C), minus4(P[(2*(i+1))%6], C))));
}

__host__ __device__ void export_bary_and_volume(ConvexCell& cc, float* out_pts, int seed) {
	float4 bary_sum = make_float4(0, 0, 0, 0);
	float cell_vol = 0; 
	float4 tet_bary;
	float tet_vol;
	float4 P[6];
	float4 C = cc.voro_seed;
	FOR(t, cc.nb_t) {
		float4 A = cc.compute_triangle_point(tr(t));
		get_tet_decomposition_of_vertex(cc, t, P);
		FOR(i, 6) {
			get_tet_volume_and_barycenter(tet_bary, tet_vol, P[i], P[(i + 1) % 6], C, A);
			bary_sum = plus4(bary_sum, mul3(tet_vol, tet_bary));
			cell_vol += tet_vol;
		}
	}
	// /cc.cell_vol
	out_pts[4 * seed] += bary_sum.x;
	out_pts[4 * seed + 1] += bary_sum.y;
	out_pts[4 * seed + 2] += bary_sum.z;
	out_pts[4 * seed + 3] += cell_vol;

}

__host__ void get_voro_diagram(ConvexCell& cc, float* out_pts, int seed, std::vector<float3>& voro_points, std::string& voro_faces) {
#ifndef __CUDA_ARCH__

	int row = voro_points.size() + 1;

	FOR(i, cc.nb_t) {
		float4 voro_vertex = cc.compute_triangle_point(tr(i));
		voro_points.push_back(make_float3(voro_vertex.x, voro_vertex.y, voro_vertex.z));
		//voro_points.push_back(make_float3(voro_vertex.x*0.9+ out_pts[3*seed]*0.1, voro_vertex.y*0.9 + out_pts[3 * seed +1] *0.1, voro_vertex.z*0.9 + out_pts[3 * seed + 2] *0.1));
	}

	std::vector<int> clipping_plane(cc.nb_v + 1, 0);
	FOR(t, cc.nb_t) {
		clipping_plane[tr(t).x]++;
		clipping_plane[tr(t).y]++;
		clipping_plane[tr(t).z]++;
	}

	std::vector<std::vector<int>> result;
	int ind = 0;

	FOR(plane, cc.nb_v) {

		if (clipping_plane[plane] > 0) {
			std::vector<int> tab_lp;
			std::vector<int> tab_v;

			FOR(tet, cc.nb_t) {

				if ((int)tr(tet).x == plane) {
					tab_v.push_back(tet);
					tab_lp.push_back(0);
				}
				else if ((int)tr(tet).y == plane) {
					tab_v.push_back(tet);
					tab_lp.push_back(1);
				}
				else if ((int)tr(tet).z == plane) {
					tab_v.push_back(tet);
					tab_lp.push_back(2);
				}
			}

			if (tab_lp.size() <= 2) {
				std::cout << (int)plane << std::endl;
			}

			int i = 0;
			int j = 0;
			result.push_back(std::vector<int>(0));

			while (result[ind].size() < tab_lp.size()) {
				int ind_i = (tab_lp[i] + 1) % 3;
				bool temp = false;
				j = 0;
				while (temp == false) {
					int ind_j = (tab_lp[j] + 2) % 3;
					if ((int)cc.ith_plane(tab_v[i], ind_i) == (int)cc.ith_plane(tab_v[j], ind_j)) {
						result[ind].push_back(tab_v[i]);
						temp = true;
						i = j;
					}
					j++;
				}
			}

			voro_faces += "f";
			FOR(i, result[ind].size()) {
				voro_faces += " ";
				voro_faces += std::to_string(row + result[ind][i] );
			}
			voro_faces += "\n";
			ind++;
		}
	}

#endif
}

__device__ __host__ float4 points_to_plane(float4 A, float4 B, float4 C) {
	float4 u = minus4(B, A);
	float4 v = minus4(C, A);
	float4 plane = cross3(u, v);
	plane.w = -dot3(plane, A);
	return make_float4(plane.x, plane.y, plane.z, plane.w);
}

__device__ __host__ void ConvexCell::clip_tet_from_points(float4 A, float4 B, float4 C, float4 D) {
	clip(nb_v) = points_to_plane(A, B, C);
	nb_v++;
	clip_by_plane(nb_v - 1);

	clip(nb_v) = points_to_plane(A, D, B);
	nb_v++;
	clip_by_plane(nb_v - 1);

	clip(nb_v) = points_to_plane(A, C, D);
	nb_v++;
	clip_by_plane(nb_v - 1);

	clip(nb_v) = points_to_plane(C, B, D);
	nb_v++;
	clip_by_plane(nb_v - 1);
}

__host__  void compute_voro_cell_CPU(
	float * pts, int nbpts, unsigned int* neigs,
	Status* gpu_stat, float* out_pts, int seed,
	std::vector<float3>& voro_points, std::string& voro_faces, 
	int tet, int* tet_indices, float* tet_pts
) {
	ConvexCell cc(seed, pts, &(gpu_stat[seed]));
	
	IF_CPU(gs.start_cell());
	float4 P0 = make_float4(tet_pts[4 * tet_indices[tet * 4]], tet_pts[4 * tet_indices[tet * 4] + 1], tet_pts[4 * tet_indices[tet * 4] + 2], tet_pts[4 * tet_indices[tet * 4] + 4]);
	float4 P1 = make_float4(tet_pts[4 * tet_indices[tet * 4 + 1]], tet_pts[4 * tet_indices[tet * 4 + 1] + 1], tet_pts[4 * tet_indices[tet * 4 + 1] + 2], tet_pts[4 * tet_indices[tet * 4 + 1] + 4]);
	float4 P2 = make_float4(tet_pts[4 * tet_indices[tet * 4 + 2]], tet_pts[4 * tet_indices[tet * 4 + 2] + 1], tet_pts[4 * tet_indices[tet * 4 + 2] + 2], tet_pts[4 * tet_indices[tet * 4 + 2] + 4]);
	float4 P3 = make_float4(tet_pts[4 * tet_indices[tet * 4 + 3]], tet_pts[4 * tet_indices[tet * 4 + 3] + 1], tet_pts[4 * tet_indices[tet * 4 + 3] + 2], tet_pts[4 * tet_indices[tet * 4 + 3] + 4]);
	cc.clip_tet_from_points(P0, P1, P2, P3);

	FOR(v, _K_) {
		unsigned int z = neigs[_K_ * seed + v];
		int cur_v = cc.new_point(z); // add new plane equation
		cc.clip_by_plane(cur_v);
		if (cc.is_security_radius_reached(point_from_ptr3(pts + 3 * z))) {
			break;
		}
		if (gpu_stat[seed] != success) {
			IF_CPU(gs.end_cell());
			return;
		}
	}

	IF_CPU(gs.end_cell());
	IF_CPU(gs.nbv[cc.nb_v]++);
	IF_CPU(gs.nbt[cc.nb_t]++);
	
	// check security radius
	if (!cc.is_security_radius_reached(point_from_ptr3(pts + 3 * neigs[_K_ * (seed + 1) - 1]))) {
		gpu_stat[seed] = security_radius_not_reached;
	}
	if (gpu_stat[seed] == success) {
		export_bary_and_volume(cc, out_pts, seed);
		get_voro_diagram(cc, out_pts, seed, voro_points, voro_faces);
	}
}

//###################  KERNEL   ######################
__host__ __device__ void compute_voro_cell(
	float * pts, int nbpts, unsigned int* neigs,
	Status* gpu_stat, float* out_pts, int seed,
	int tet, int* tet_indices, float* tet_pts
) {
    ConvexCell cc(seed, pts, &(gpu_stat[seed]));

	//clip by tet
	float4 P0 = make_float4(tet_pts[4 * tet_indices[tet * 4]], tet_pts[4 * tet_indices[tet * 4] + 1], tet_pts[4 * tet_indices[tet * 4] + 2], tet_pts[4 * tet_indices[tet * 4] + 4]);
	float4 P1 = make_float4(tet_pts[4 * tet_indices[tet * 4 + 1]], tet_pts[4 * tet_indices[tet * 4 + 1] + 1], tet_pts[4 * tet_indices[tet * 4 + 1] + 2], tet_pts[4 * tet_indices[tet * 4 + 1] + 4]);
	float4 P2 = make_float4(tet_pts[4 * tet_indices[tet * 4 + 2]], tet_pts[4 * tet_indices[tet * 4 + 2] + 1], tet_pts[4 * tet_indices[tet * 4 + 2] + 2], tet_pts[4 * tet_indices[tet * 4 + 2] + 4]);
	float4 P3 = make_float4(tet_pts[4 * tet_indices[tet * 4 + 3]], tet_pts[4 * tet_indices[tet * 4 + 3] + 1], tet_pts[4 * tet_indices[tet * 4 + 3] + 2], tet_pts[4 * tet_indices[tet * 4 + 3] + 4]);
	cc.clip_tet_from_points(P0, P1, P2, P3);

    FOR(v, _K_) {
		unsigned int z = neigs[_K_ * seed + v];
		int cur_v = cc.new_point(z);
		cc.clip_by_plane(cur_v);
		if (cc.is_security_radius_reached(point_from_ptr3(pts + 3*z))) {
			break;
		}
		if (gpu_stat[seed] != success) {
			return;
		}
    }
    // check security radius
    if (!cc.is_security_radius_reached(point_from_ptr3(pts + 3 * neigs[_K_ * (seed+1) -1]))) {
        gpu_stat[seed] = security_radius_not_reached;
    }
	if (gpu_stat[seed] == success) {
		export_bary_and_volume(cc, out_pts, seed);
	}
}

//----------------------------------KERNEL
__global__ void voro_cell_test_GPU_param(float * pts, int nbpts, int nbtets, unsigned int* neigs, Status* gpu_stat, float* out_pts, int* tet_indices, float* tet_pts) {

	int seed = blockIdx.x * blockDim.x + threadIdx.x;

	if (seed < nbpts){
		for (int tet = 0; tet < nbtets; ++tet) {
			compute_voro_cell(pts, nbpts, neigs, gpu_stat, out_pts, seed, tet, tet_indices, tet_pts);
		}
	}
}

//----------------------------------WRAPPER
template <class T> struct GPUBuffer {
    void init(T* data) {
        IF_VERBOSE(std::cerr << "GPU: " << size * sizeof(T)/1048576 << " Mb used" << std::endl);
        cpu_data = data;
        cuda_check(cudaMalloc((void**)& gpu_data, size * sizeof(T)));
        cpu2gpu();
    }
    GPUBuffer(std::vector<T>& v) {size = v.size();init(v.data());}
    ~GPUBuffer() { cuda_check(cudaFree(gpu_data)); }

    void cpu2gpu() { cuda_check(cudaMemcpy(gpu_data, cpu_data, size * sizeof(T), cudaMemcpyHostToDevice)); }
    void gpu2cpu() { cuda_check(cudaMemcpy(cpu_data, gpu_data, size * sizeof(T), cudaMemcpyDeviceToHost)); }

    T* cpu_data;
    T* gpu_data;
    int size;
};

char StatusStr[6][128] = {
    "triangle_overflow","vertex_overflow","inconsistent_boundary","security_radius_not_reached","success", "needs_exact_predicates"
};

void show_status_stats(std::vector<Status> &stat) {
    IF_VERBOSE(std::cerr << " \n\n\n---------Summary of success/failure------------\n");
    std::vector<int> nb_statuss(6, 0);
    FOR(i, stat.size()) nb_statuss[stat[i]]++;
    IF_VERBOSE(FOR(r, 6) std::cerr << " " << StatusStr[r] << "   " << nb_statuss[r] << "\n";)
        std::cerr << " " << StatusStr[4] << "   " << nb_statuss[4] << " /  " << stat.size() << "\n";
}

void cuda_check_error() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { fprintf(stderr, "Failed (1) (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); }
}

void compute_voro_diagram_GPU(
    std::vector<float>& pts, std::vector<Status> &stat, std::vector<float>& bary,
	std::vector<float>& tet_pts, std::vector<int>& tet_indices, std::vector<int>* KNN, int nb_Lloyd_iter
) {
    int nbpts = pts.size() / 3;
    kn_problem *kn = NULL;
    {
        IF_VERBOSE(Stopwatch W("GPU KNN"));
        kn = kn_prepare((float3*) pts.data(), nbpts);
        cudaMemcpy(pts.data(), kn->d_stored_points, kn->allocated_points * sizeof(float) * 3, cudaMemcpyDeviceToHost);
        cuda_check_error();
        kn_solve(kn);
        IF_VERBOSE(kn_print_stats(kn));
    }

	int nbtets = tet_pts.size() / 3; 
    GPUBuffer<float> out_pts_w(bary);
	GPUBuffer<int> tet_indices_w(tet_indices);
	GPUBuffer<float> tet_pts_w(tet_pts);
    GPUBuffer<Status> gpu_stat(stat);

//  if (nb_Lloyd_iter == 0) {
        IF_VERBOSE(Stopwatch W("GPU voro kernel only"));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

		//dim3 threads_per_block(16, 16, 1); // A 16 x 16 block threads
		//dim3 number_of_blocks((nbpts / threads_per_block.x) + 1, (nbtets / threads_per_block.y) + 1, 1);

        voro_cell_test_GPU_param <<< (nbpts / 16) + 1, 16 >>> ((float*)kn->d_stored_points, nbpts, nbtets, kn->d_knearests, gpu_stat.gpu_data, out_pts_w.gpu_data, tet_indices_w.gpu_data, tet_pts_w.gpu_data);
		
		cuda_check_error();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        IF_VERBOSE(std::cerr << "kn voro: " << milliseconds << " msec" << std::endl);
		/*
//  }

//  // Lloyd
//  FOR(lit,nb_Lloyd_iter){
//      IF_VERBOSE(Stopwatch W("Loyd iterations"));
//      cudaEvent_t start, stop;
//      cudaEventCreate(&start);
//      cudaEventCreate(&stop);
//      cudaEventRecord(start);

//      voro_cell_test_GPU_param << < nbpts / VORO_BLOCK_SIZE + 1, VORO_BLOCK_SIZE >> > ((float*)kn->d_stored_points, nbpts, kn->d_knearests, gpu_stat.gpu_data, out_pts_w.gpu_data);
//      cuda_check_error();

//      voro_cell_test_GPU_param << < nbpts / VORO_BLOCK_SIZE + 1, VORO_BLOCK_SIZE >> > (out_pts_w.gpu_data, nbpts, kn->d_knearests, gpu_stat.gpu_data, (float*)kn->d_stored_points);
//      cuda_check_error();


//      cudaEventRecord(stop);
//      cudaEventSynchronize(stop);
//      float milliseconds = 0;
//      cudaEventElapsedTime(&milliseconds, start, stop);
//      IF_VERBOSE(std::cerr << "kn voro: " << milliseconds << " msec" << std::endl);
//  }
*/

		
    {
        IF_VERBOSE(Stopwatch W("copy data back to the cpu"));
        out_pts_w.gpu2cpu();
        gpu_stat.gpu2cpu();
    }

    // Read back nearest neighbor indices if KNN is specified.
    if(KNN != NULL) {
	KNN->resize(kn->allocated_points*_K_);
	cuda_check(cudaMemcpy(
             KNN->data(), kn->d_knearests, sizeof(int) * _K_ * kn->allocated_points, cudaMemcpyDeviceToHost)
	);	
    }
	
    kn_free(&kn);
    show_status_stats(stat);
}

void compute_voro_diagram_CPU(
	std::vector<float>& pts, std::vector<Status>& stat, std::vector<float>& bary,
	std::vector<float>& tet_pts, std::vector<int>& tet_indices, std::vector<int>* KNN,
	int nb_Lloyd_iter
) {
	
	int nbpts = pts.size() / 3;
	kn_problem *kn = NULL;
	{
		IF_VERBOSE(Stopwatch W("GPU KNN"));
		kn = kn_prepare((float3*)pts.data(), nbpts);
		kn_solve(kn);
		IF_VERBOSE(kn_print_stats(kn));
	}

	float* nvpts = (float*)kn_get_points(kn);
	unsigned int* knn = kn_get_knearests(kn);

	IF_VERBOSE(Stopwatch W("CPU VORO KERNEL"));
	
	std::vector<float3> voro_points;
	std::string voro_faces;
	int nbtet = tet_indices.size() / 4;

	FOR(seed, nbpts) {
		FOR(tet, nbtet) {
			compute_voro_cell_CPU(
				nvpts, nbpts, knn, stat.data(), bary.data(), seed, voro_points, voro_faces, tet, tet_indices.data(), tet_pts.data()
			);
		}
	}
	std::fstream output_file;
	output_file.open("voro_tet.obj", std::ios_base::out);
	output_file << "# voro_cell.obj" << std::endl;
	output_file << "#" << std::endl;
	output_file << std::endl;
	output_file << "o voro_cell" << std::endl;
	output_file << std::endl;
	FOR(i, voro_points.size()) {
		output_file << "v " << voro_points[i].x << " " << voro_points[i].y << " " << voro_points[i].z << std::endl;
	}
	output_file << std::endl;
	output_file << voro_faces << std::endl;
	output_file.close();
	
	static int callid = 0;
	/*
	//FOR(i, nb_Lloyd_iter) {
	//	gs.reset();
	//	FOR(seed, nbpts)  compute_voro_cell(nvpts, nbpts, knn, stat.data(), bary.data(), seed);
	//	FOR(i, pts.size()) pts[i] = bary[i];
	//}
	*/
	callid++;

	if (KNN != NULL) {
		FOR(i, _K_ * kn->allocated_points) {
			KNN->push_back(knn[i]);
		}
	}
	kn_free(&kn);
	free(nvpts);
	free(knn);
	show_status_stats(stat);
}

