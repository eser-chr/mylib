// CUDA port of AtomsCPU (see atoms_cpu.hpp). Same physics, same neighbour-list
// criterion; the differences are GPU-motivated:
//   * one thread per atom everywhere;
//   * FULL neighbour list (i sees j and j sees i), so forces need no atomics;
//   * neighbour list stored "column major" nb[k*N + i] for coalesced loads;
//   * cell list built by counting sort (atomicAdd + exclusive scan) instead of a linked list;
//   * Langevin noise from a per-atom counter-based Philox generator (cuRAND).
#include "atoms_gpu.cuh"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>

#include <string>

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess)                                                      \
            throw std::runtime_error(std::string("CUDA error: ") +                    \
                                     cudaGetErrorString(err_) + " at " __FILE__ ":" + \
                                     std::to_string(__LINE__));                       \
    } while (0)
#define CUDA_LAUNCH_CHECK() CUDA_CHECK(cudaGetLastError())

namespace {

constexpr int BLOCK = 256;
inline int nblocks(std::size_t n) { return static_cast<int>((n + BLOCK - 1) / BLOCK); }

using RngState = curandStatePhilox4_32_10_t;

struct Box {
    double L[3], invL[3];
    int periodic;
};

// ------------------------------------------------------------------ helpers

__device__ __forceinline__ void min_image(double& dx, double& dy, double& dz, const Box& b) {
    if (b.periodic) {
        dx -= b.L[0] * rint(dx * b.invL[0]);
        dy -= b.L[1] * rint(dy * b.invL[1]);
        dz -= b.L[2] * rint(dz * b.invL[2]);
    }
}

// Order-preserving map double -> uint64, so atomicMin/atomicMax on integers
// give min/max of doubles (there is no native atomicMax for double).
__host__ __device__ inline unsigned long long ord_encode(double x) {
    unsigned long long b;
    memcpy(&b, &x, sizeof b);
    return (b >> 63) ? ~b : (b | 0x8000000000000000ULL);
}
__host__ __device__ inline double ord_decode(unsigned long long b) {
    b = (b >> 63) ? (b & 0x7FFFFFFFFFFFFFFFULL) : ~b;
    double x;
    memcpy(&x, &b, sizeof x);
    return x;
}

template <bool IsMax>
__device__ double block_reduce(double v) {
    __shared__ double s[BLOCK / 32];
    auto op = [](double a, double c) { return IsMax ? fmax(a, c) : fmin(a, c); };
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    for (int o = 16; o > 0; o >>= 1) v = op(v, __shfl_down_sync(0xffffffffu, v, o));
    if (lane == 0) s[wid] = v;
    __syncthreads();
    if (wid == 0) {
        v = (lane < BLOCK / 32) ? s[lane] : (IsMax ? -INFINITY : INFINITY);
        for (int o = 16; o > 0; o >>= 1) v = op(v, __shfl_down_sync(0xffffffffu, v, o));
    }
    __syncthreads();   // s[] may be reused by the next call
    return v;          // valid in thread 0
}

__device__ __forceinline__ int cell_coord(double x, const CellGrid& g, int d) {
    const int c = static_cast<int>(floor((x - g.lo[d]) * g.inv[d]));
    return min(max(c, 0), g.n[d] - 1);
}

// ------------------------------------------------------------------ kernels

__global__ void k_init_rng(int N, unsigned long long seed, RngState* st) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) curand_init(seed, i, 0, &st[i]);   // subsequence i -> independent stream per atom
}

// B (+A, or A-O-A for Langevin), then track max displacement since last list build.
__global__ void k_first_half(int N, double* __restrict__ pos, double* __restrict__ vel,
                             const double* __restrict__ f, const double* __restrict__ pos0,
                             double h, double dt, int langevin, double c1, double c2,
                             RngState* __restrict__ st, unsigned long long* maxdisp) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    double d2 = 0.0;
    if (i < N) {
        double x[3], v[3];
        for (int d = 0; d < 3; ++d) {
            x[d] = pos[3*i + d];
            v[d] = vel[3*i + d] + h * f[3*i + d];                          // B
        }
        if (langevin) {
            RngState s = st[i];
            const double2 g01 = curand_normal2_double(&s);
            const double  g[3] = {g01.x, g01.y, curand_normal_double(&s)};
            st[i] = s;
            for (int d = 0; d < 3; ++d) {
                x[d] += 0.5 * dt * v[d];                                   // A
                v[d]  = c1 * v[d] + c2 * g[d];                             // O
                x[d] += 0.5 * dt * v[d];                                   // A
            }
        } else {
            for (int d = 0; d < 3; ++d) x[d] += dt * v[d];
        }
        for (int d = 0; d < 3; ++d) {
            pos[3*i + d] = x[d];
            vel[3*i + d] = v[d];
            if (pos0) { const double u = x[d] - pos0[3*i + d]; d2 += u * u; }
        }
    }
    if (pos0) {   // uniform branch: every thread of the block takes it
        d2 = block_reduce<true>(d2);
        if (threadIdx.x == 0) atomicMax(maxdisp, ord_encode(d2));
    }
}

__global__ void k_second_half(int n3, double* __restrict__ vel, const double* __restrict__ f, double h) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n3) vel[k] += h * f[k];
}

// Bounding box (open boundaries). out[0..2] = min, out[3..5] = max, ord-encoded.
__global__ void k_bbox(int N, const double* __restrict__ pos, unsigned long long* out) {
    double lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY};
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x)
        for (int d = 0; d < 3; ++d) {
            lo[d] = fmin(lo[d], pos[3*i + d]);
            hi[d] = fmax(hi[d], pos[3*i + d]);
        }
    for (int d = 0; d < 3; ++d) {
        const double a = block_reduce<false>(lo[d]);
        const double b = block_reduce<true>(hi[d]);
        if (threadIdx.x == 0) {
            atomicMin(&out[d], ord_encode(a));
            atomicMax(&out[3 + d], ord_encode(b));
        }
    }
}

// Wrap into the box, remember reference positions, bin into cells.
__global__ void k_prepare_build(int N, double* __restrict__ pos, double* __restrict__ pos0, Box box,
                                CellGrid g, int* __restrict__ cell_of, int* __restrict__ cell_count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int c[3];
    for (int d = 0; d < 3; ++d) {
        double x = pos[3*i + d];
        if (box.periodic) { x -= box.L[d] * floor(x * box.invL[d]); pos[3*i + d] = x; }
        pos0[3*i + d] = x;
        c[d] = cell_coord(x, g, d);
    }
    const int cell = (c[2] * g.n[1] + c[1]) * g.n[0] + c[0];
    cell_of[i] = cell;
    atomicAdd(&cell_count[cell], 1);
}

__global__ void k_fill_cells(int N, const int* __restrict__ cell_of, int* __restrict__ cursor,
                             int* __restrict__ cell_atoms) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) cell_atoms[atomicAdd(&cursor[cell_of[i]], 1)] = i;
}

__device__ __forceinline__ void try_add(int i, int j, const double* __restrict__ pos, const Box& box,
                                        double rl2, int N, int max_nb, int* __restrict__ nb, int& cnt,
                                        double xi, double yi, double zi) {
    if (j == i) return;
    double dx = xi - pos[3*j], dy = yi - pos[3*j + 1], dz = zi - pos[3*j + 2];
    min_image(dx, dy, dz, box);
    if (dx*dx + dy*dy + dz*dz < rl2) {
        if (cnt < max_nb) nb[cnt * N + i] = j;
        ++cnt;
    }
}

__global__ void k_build_nb_cells(int N, const double* __restrict__ pos, Box box, CellGrid g,
                                 const int* __restrict__ cell_start, const int* __restrict__ cell_count,
                                 const int* __restrict__ cell_atoms, double rl2, int max_nb,
                                 int* __restrict__ nb, int* __restrict__ nb_count, int* overflow) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const double xi = pos[3*i], yi = pos[3*i + 1], zi = pos[3*i + 2];
    const int ci[3] = {cell_coord(xi, g, 0), cell_coord(yi, g, 1), cell_coord(zi, g, 2)};
    int cnt = 0;
    for (int oz = -1; oz <= 1; ++oz)
    for (int oy = -1; oy <= 1; ++oy)
    for (int ox = -1; ox <= 1; ++ox) {
        int c[3] = {ci[0] + ox, ci[1] + oy, ci[2] + oz};
        bool skip = false;
        for (int d = 0; d < 3; ++d) {
            if (box.periodic)                      c[d] = (c[d] + g.n[d]) % g.n[d];
            else if (c[d] < 0 || c[d] >= g.n[d]) skip = true;
        }
        if (skip) continue;
        const int cell = (c[2] * g.n[1] + c[1]) * g.n[0] + c[0];
        const int s = cell_start[cell], e = s + cell_count[cell];
        for (int k = s; k < e; ++k)
            try_add(i, cell_atoms[k], pos, box, rl2, N, max_nb, nb, cnt, xi, yi, zi);
    }
    nb_count[i] = cnt;
    if (cnt > max_nb) atomicMax(overflow, cnt);
}

__global__ void k_build_nb_allpairs(int N, const double* __restrict__ pos, Box box, double rl2, int max_nb,
                                    int* __restrict__ nb, int* __restrict__ nb_count, int* overflow) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const double xi = pos[3*i], yi = pos[3*i + 1], zi = pos[3*i + 2];
    int cnt = 0;
    for (int j = 0; j < N; ++j) try_add(i, j, pos, box, rl2, N, max_nb, nb, cnt, xi, yi, zi);
    nb_count[i] = cnt;
    if (cnt > max_nb) atomicMax(overflow, cnt);
}

struct LJ {
    double rc2, s2, eps;
    // Adds the force of j on i; returns the pair energy (0 outside the cutoff).
    __device__ __forceinline__ double operator()(double dx, double dy, double dz,
                                                 double& fx, double& fy, double& fz) const {
        const double r2 = dx*dx + dy*dy + dz*dz;
        if (r2 > rc2) return 0.0;
        const double sr2  = s2 / r2;
        const double sr6  = sr2 * sr2 * sr2;
        const double sr12 = sr6 * sr6;
        const double f = 24.0 * eps * (2.0 * sr12 - sr6) / r2;
        fx += f * dx; fy += f * dy; fz += f * dz;
        return 4.0 * eps * (sr12 - sr6);
    }
};

__global__ void k_forces_nb(int N, const double* __restrict__ pos, Box box, LJ lj,
                            const int* __restrict__ nb, const int* __restrict__ nb_count,
                            double* __restrict__ f, double* __restrict__ epot) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const double xi = pos[3*i], yi = pos[3*i + 1], zi = pos[3*i + 2];
    double fx = 0.0, fy = 0.0, fz = 0.0, e = 0.0;
    const int n = nb_count[i];
    for (int k = 0; k < n; ++k) {
        const int j = nb[k * N + i];
        double dx = xi - pos[3*j], dy = yi - pos[3*j + 1], dz = zi - pos[3*j + 2];
        min_image(dx, dy, dz, box);
        e += lj(dx, dy, dz, fx, fy, fz);
    }
    f[3*i] = fx; f[3*i + 1] = fy; f[3*i + 2] = fz;
    epot[i] = 0.5 * e;   // each pair is counted twice with a full list
}

__global__ void k_forces_allpairs(int N, const double* __restrict__ pos, Box box, LJ lj,
                                  double* __restrict__ f, double* __restrict__ epot) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const double xi = pos[3*i], yi = pos[3*i + 1], zi = pos[3*i + 2];
    double fx = 0.0, fy = 0.0, fz = 0.0, e = 0.0;
    for (int j = 0; j < N; ++j) {   // all threads read the same j -> broadcast load
        if (j == i) continue;
        double dx = xi - pos[3*j], dy = yi - pos[3*j + 1], dz = zi - pos[3*j + 2];
        min_image(dx, dy, dz, box);
        e += lj(dx, dy, dz, fx, fy, fz);
    }
    f[3*i] = fx; f[3*i + 1] = fy; f[3*i + 2] = fz;
    epot[i] = 0.5 * e;
}

struct Square {
    __host__ __device__ double operator()(double v) const { return v * v; }
};

Box make_box(const MDParams& p) {
    if (!p.periodic()) return Box{{0, 0, 0}, {0, 0, 0}, 0};
    return Box{{p.box[0], p.box[1], p.box[2]}, {1.0 / p.box[0], 1.0 / p.box[1], 1.0 / p.box[2]}, 1};
}

template <class T>
void dev_alloc(T*& ptr, std::size_t n) {
    if (ptr) CUDA_CHECK(cudaFree(ptr));
    ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
}

}  // namespace

// ====================================================================== AtomsGPU

AtomsGPU::AtomsGPU(const std::vector<double>& pos, const MDParams& params)
    : N(pos.size() / 3), p(params) {
    if (pos.size() % 3 != 0 || N < 2) throw std::invalid_argument("need 3N positions, N >= 2");
    p.validate();

    dev_alloc(d_pos, 3 * N);
    dev_alloc(d_vel, 3 * N);
    dev_alloc(d_f, 3 * N);
    dev_alloc(d_pos0, 3 * N);
    dev_alloc(d_epot, N);
    dev_alloc(d_u64, 6);
    dev_alloc(d_cell_of, N);
    dev_alloc(d_overflow, 1);
    dev_alloc(d_nb_count, N);
    RngState* rng = nullptr;
    dev_alloc(rng, N);
    d_rng = rng;

    CUDA_CHECK(cudaMemcpy(d_pos, pos.data(), 3 * N * sizeof(double), cudaMemcpyHostToDevice));

    // Same initial velocities as AtomsCPU for the same seed.
    std::mt19937_64 host_rng(p.seed);
    set_velocities(maxwell_boltzmann(N, p.kT, p.mass, host_rng));

    k_init_rng<<<nblocks(N), BLOCK>>>(static_cast<int>(N), p.seed, rng);
    CUDA_LAUNCH_CHECK();

    if (p.use_nlist) {
        // Initial guess for neighbours per atom; grown automatically on overflow.
        double guess = 64.0;
        if (p.periodic()) {
            const double rho = N / (p.box[0] * p.box[1] * p.box[2]);
            const double rl = p.rlist();
            guess = 1.3 * (4.0 / 3.0) * M_PI * rl * rl * rl * rho + 16.0;
        }
        max_neighbors = std::min(static_cast<int>(guess), static_cast<int>(N));
        build_nlist();
    }
    compute_forces();
}

AtomsGPU::~AtomsGPU() {
    // No throwing from a destructor.
    for (void* ptr : {(void*)d_pos, (void*)d_vel, (void*)d_f, (void*)d_pos0, (void*)d_epot, (void*)d_u64,
                      (void*)d_cell_of, (void*)d_cell_count, (void*)d_cell_start, (void*)d_cell_cursor,
                      (void*)d_cell_atoms, (void*)d_nb, (void*)d_nb_count, (void*)d_overflow, d_rng})
        if (ptr) cudaFree(ptr);
}

void AtomsGPU::build_nlist() {
    ++n_builds;
    const int n = static_cast<int>(N);
    const Box box = make_box(p);

    double lo[3] = {0, 0, 0}, hi[3] = {0, 0, 0};
    if (!p.periodic()) {
        unsigned long long init[6];
        for (int d = 0; d < 3; ++d) { init[d] = ord_encode(INFINITY); init[3 + d] = ord_encode(-INFINITY); }
        CUDA_CHECK(cudaMemcpy(d_u64, init, sizeof init, cudaMemcpyHostToDevice));
        k_bbox<<<std::min(nblocks(N), 1024), BLOCK>>>(n, d_pos, d_u64);
        CUDA_LAUNCH_CHECK();
        CUDA_CHECK(cudaMemcpy(init, d_u64, sizeof init, cudaMemcpyDeviceToHost));
        for (int d = 0; d < 3; ++d) { lo[d] = ord_decode(init[d]); hi[d] = ord_decode(init[3 + d]); }
    }
    const CellGrid g = make_grid(p, lo, hi, N);
    const int nc = g.ncells();
    if (nc > ncells_alloc) {
        dev_alloc(d_cell_count, nc);
        dev_alloc(d_cell_start, nc);
        dev_alloc(d_cell_cursor, nc);
        if (!d_cell_atoms) dev_alloc(d_cell_atoms, N);
        ncells_alloc = nc;
    }

    // Counting sort of atoms into cells
    CUDA_CHECK(cudaMemset(d_cell_count, 0, nc * sizeof(int)));
    k_prepare_build<<<nblocks(N), BLOCK>>>(n, d_pos, d_pos0, box, g, d_cell_of, d_cell_count);
    CUDA_LAUNCH_CHECK();
    thrust::exclusive_scan(thrust::device, d_cell_count, d_cell_count + nc, d_cell_start);
    CUDA_CHECK(cudaMemcpy(d_cell_cursor, d_cell_start, nc * sizeof(int), cudaMemcpyDeviceToDevice));
    k_fill_cells<<<nblocks(N), BLOCK>>>(n, d_cell_of, d_cell_cursor, d_cell_atoms);
    CUDA_LAUNCH_CHECK();

    // Neighbour list; retry with a bigger buffer on overflow
    const double rl2 = p.rlist() * p.rlist();
    while (true) {
        if (max_neighbors > nb_alloc) {
            dev_alloc(d_nb, static_cast<std::size_t>(max_neighbors) * N);
            nb_alloc = max_neighbors;
        }
        CUDA_CHECK(cudaMemset(d_overflow, 0, sizeof(int)));
        if (g.ok)
            k_build_nb_cells<<<nblocks(N), BLOCK>>>(n, d_pos, box, g, d_cell_start, d_cell_count,
                                                    d_cell_atoms, rl2, max_neighbors, d_nb, d_nb_count,
                                                    d_overflow);
        else
            k_build_nb_allpairs<<<nblocks(N), BLOCK>>>(n, d_pos, box, rl2, max_neighbors, d_nb, d_nb_count,
                                                       d_overflow);
        CUDA_LAUNCH_CHECK();
        int overflow = 0;
        CUDA_CHECK(cudaMemcpy(&overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost));
        if (overflow == 0) break;
        max_neighbors = overflow + overflow / 5 + 8;
    }
}

void AtomsGPU::compute_forces() {
    const Box box = make_box(p);
    const LJ lj{p.rcut * p.rcut, p.sigma * p.sigma, p.epsilon};
    if (p.use_nlist)
        k_forces_nb<<<nblocks(N), BLOCK>>>(static_cast<int>(N), d_pos, box, lj, d_nb, d_nb_count, d_f, d_epot);
    else
        k_forces_allpairs<<<nblocks(N), BLOCK>>>(static_cast<int>(N), d_pos, box, lj, d_f, d_epot);
    CUDA_LAUNCH_CHECK();
}

void AtomsGPU::first_half(bool langevin) {
    const double h  = 0.5 * p.dt / p.mass;
    const double c1 = std::exp(-p.gamma * p.dt);
    const double c2 = std::sqrt((1.0 - c1 * c1) * p.kT / p.mass);
    if (p.use_nlist) CUDA_CHECK(cudaMemsetAsync(d_u64, 0, sizeof(unsigned long long)));
    k_first_half<<<nblocks(N), BLOCK>>>(static_cast<int>(N), d_pos, d_vel, d_f, p.use_nlist ? d_pos0 : nullptr,
                                        h, p.dt, langevin ? 1 : 0, c1, c2,
                                        static_cast<RngState*>(d_rng), d_u64);
    CUDA_LAUNCH_CHECK();
}

void AtomsGPU::maybe_rebuild() {
    if (!p.use_nlist) return;
    // The only host<->device sync per step: one 8-byte copy.
    unsigned long long enc = 0;
    CUDA_CHECK(cudaMemcpy(&enc, d_u64, sizeof enc, cudaMemcpyDeviceToHost));
    if (ord_decode(enc) > 0.25 * p.skin * p.skin) build_nlist();
}

void AtomsGPU::second_half() {
    k_second_half<<<nblocks(3 * N), BLOCK>>>(static_cast<int>(3 * N), d_vel, d_f, 0.5 * p.dt / p.mass);
    CUDA_LAUNCH_CHECK();
}

void AtomsGPU::verlet_step() {
    first_half(false);
    maybe_rebuild();
    compute_forces();
    second_half();
}

void AtomsGPU::langevin_step() {
    first_half(true);
    maybe_rebuild();
    compute_forces();
    second_half();
}

void AtomsGPU::run(int nsteps, bool langevin) {
    for (int s = 0; s < nsteps; ++s) langevin ? langevin_step() : verlet_step();
    synchronize();
}

void AtomsGPU::synchronize() const { CUDA_CHECK(cudaDeviceSynchronize()); }

void AtomsGPU::set_velocities(const std::vector<double>& v) {
    if (v.size() != 3 * N) throw std::invalid_argument("velocities must have 3N entries");
    CUDA_CHECK(cudaMemcpy(d_vel, v.data(), 3 * N * sizeof(double), cudaMemcpyHostToDevice));
}

static std::vector<double> download(const double* d, std::size_t n) {
    std::vector<double> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(double), cudaMemcpyDeviceToHost));
    return h;
}

std::vector<double> AtomsGPU::get_positions() const  { return download(d_pos, 3 * N); }
std::vector<double> AtomsGPU::get_velocities() const { return download(d_vel, 3 * N); }
std::vector<double> AtomsGPU::get_forces() const     { return download(d_f, 3 * N); }

double AtomsGPU::epot() const { return thrust::reduce(thrust::device, d_epot, d_epot + N, 0.0); }

double AtomsGPU::ekin() const {
    return 0.5 * p.mass *
           thrust::transform_reduce(thrust::device, d_vel, d_vel + 3 * N, Square{}, 0.0, thrust::plus<double>());
}
