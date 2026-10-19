#pragma once
// GPU engine. The header deliberately contains no CUDA types so it can be
// included from host-only translation units (main.cpp, bindings.cpp) compiled by g++.
#include "common.hpp"

class AtomsGPU {
public:
    AtomsGPU(const std::vector<double>& pos, const MDParams& params);
    ~AtomsGPU();
    AtomsGPU(const AtomsGPU&) = delete;
    AtomsGPU& operator=(const AtomsGPU&) = delete;

    void verlet_step();
    void langevin_step();
    void run(int nsteps, bool langevin);

    void set_velocities(const std::vector<double>& v);
    std::vector<double> get_positions() const;
    std::vector<double> get_velocities() const;
    std::vector<double> get_forces() const;

    double epot() const;   // reductions on the device, done on demand
    double ekin() const;
    double temperature() const { return 2.0 * ekin() / (3.0 * N); }
    void synchronize() const;

    std::size_t N;
    MDParams p;
    long n_builds = 0;
    int max_neighbors = 0;

private:
    double *d_pos = nullptr, *d_vel = nullptr, *d_f = nullptr, *d_pos0 = nullptr, *d_epot = nullptr;
    unsigned long long* d_u64 = nullptr;      // scratch for atomic min/max (6 slots)
    int *d_cell_of = nullptr, *d_cell_count = nullptr, *d_cell_start = nullptr,
        *d_cell_cursor = nullptr, *d_cell_atoms = nullptr;
    int *d_nb = nullptr, *d_nb_count = nullptr, *d_overflow = nullptr;
    void* d_rng = nullptr;                    // curandStatePhilox4_32_10_t[N]
    int ncells_alloc = 0;
    int nb_alloc = 0;

    void build_nlist();
    void maybe_rebuild();
    void compute_forces();
    void first_half(bool langevin);
    void second_half();
};
