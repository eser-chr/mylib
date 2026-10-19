#pragma once
// Pieces shared by the CPU and GPU engines (host-side only).
#include <array>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <stdexcept>

// Parameters, reduced LJ units, k_B = 1.
struct MDParams {
    double kT      = 1.0;
    double mass    = 1.0;
    double dt      = 0.005;
    double gamma   = 1.0;     // Langevin friction
    double epsilon = 1.0;
    double sigma   = 1.0;
    double rcut    = 2.5;
    double skin    = 0.3;     // Verlet-list skin; rebuild when max displacement > skin/2
    unsigned long long seed = 42;
    bool   use_nlist = true;  // false -> O(N^2) all-pairs forces

    // Box lengths. All > 0 -> periodic (minimum image). All 0 -> open boundaries.
    std::array<double, 3> box = {0.0, 0.0, 0.0};

    bool periodic() const { return box[0] > 0.0 && box[1] > 0.0 && box[2] > 0.0; }
    double rlist() const { return rcut + skin; }

    void validate() const {
        if (periodic()) {
            const double r = use_nlist ? rlist() : rcut;
            for (double L : box)
                if (r > 0.5 * L)
                    throw std::invalid_argument("cutoff (+skin) must be <= box/2 for minimum image");
        } else if (box[0] != 0.0 || box[1] != 0.0 || box[2] != 0.0) {
            throw std::invalid_argument("box must be all > 0 (periodic) or all 0 (open)");
        }
    }
};

// Uniform cell grid used to build the neighbour list.
// Plain-old-data so it can be passed by value to CUDA kernels.
struct CellGrid {
    int    n[3];
    double lo[3];
    double inv[3];     // cells per unit length
    int    ok;         // 0 -> too few cells, fall back to all-pairs list build

    int ncells() const { return n[0] * n[1] * n[2]; }
};

// lo/hi: bounding box of the atoms (ignored when periodic).
inline CellGrid make_grid(const MDParams& p, const double lo[3], const double hi[3], std::size_t N) {
    CellGrid g{};
    g.ok = 1;
    const double rl = p.rlist();
    // Cap on cells per dimension for open systems, so an evaporated atom
    // far away cannot blow up the grid. Larger cells are still correct.
    const int cap = std::max(1, static_cast<int>(std::cbrt(2.0 * N)) + 1);
    for (int d = 0; d < 3; ++d) {
        if (p.periodic()) {
            g.lo[d] = 0.0;
            g.n[d]  = static_cast<int>(std::floor(p.box[d] / rl));
            if (g.n[d] < 3) g.ok = 0;   // 27-cell stencil would visit a cell twice
            g.n[d]  = std::max(g.n[d], 1);
            g.inv[d] = g.n[d] / p.box[d];
        } else {
            const double ext = hi[d] - lo[d];
            g.lo[d] = lo[d];
            g.n[d]  = std::clamp(static_cast<int>(std::floor(ext / rl)), 1, cap);
            g.inv[d] = ext > 0.0 ? g.n[d] / ext : 0.0;
        }
    }
    return g;
}

// Maxwell-Boltzmann velocities with the centre-of-mass drift removed.
// Used by both engines so that the same seed gives the same initial state.
inline std::vector<double> maxwell_boltzmann(std::size_t N, double kT, double mass, std::mt19937_64& rng) {
    std::normal_distribution<double> gauss(0.0, 1.0);
    std::vector<double> vel(3 * N);
    const double s = std::sqrt(kT / mass);
    for (auto& v : vel) v = s * gauss(rng);
    for (int d = 0; d < 3; ++d) {
        double vc = 0.0;
        for (std::size_t i = 0; i < N; ++i) vc += vel[3*i + d];
        vc /= static_cast<double>(N);
        for (std::size_t i = 0; i < N; ++i) vel[3*i + d] -= vc;
    }
    return vel;
}
