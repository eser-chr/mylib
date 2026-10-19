#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>

#include "atoms_cpu.hpp"
#ifdef WITH_CUDA
#include "atoms_gpu.cuh"
#endif

// n x n x n simple cubic lattice with spacing a
static std::vector<double> cubic_lattice(int n, double a) {
    std::vector<double> x;
    x.reserve(3 * n * n * n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k) {
                x.push_back((i + 0.5) * a);
                x.push_back((j + 0.5) * a);
                x.push_back((k + 0.5) * a);
            }
    return x;
}

template <class Engine>
static double seconds_per_step(Engine& e, int nsteps) {
    const auto t0 = std::chrono::steady_clock::now();
    e.run(nsteps, /*langevin=*/true);
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count() / nsteps;
}

int main(int argc, char** argv) {
    // --- 1) The original demo: 4x4x4 open cluster, NVE then Langevin -------------
    {
        MDParams p;
        p.kT = 0.5;
        AtomsCPU atoms(cubic_lattice(4, 1.2), p);

        std::cout << "NVE (CPU, neighbour list)\n";
        for (int s = 0; s <= 1000; ++s) {
            if (s % 100 == 0)
                std::cout << s << "  Etot = " << atoms.epot + atoms.ekin()
                          << "  T = " << atoms.temperature() << "\n";
            atoms.verlet_step();
        }
        std::cout << "Langevin\n";
        for (int s = 0; s <= 5000; ++s) {
            if (s % 500 == 0)
                std::cout << s << "  Epot = " << atoms.epot << "  T = " << atoms.temperature() << "\n";
            atoms.langevin_step();
        }
        std::cout << "neighbour-list builds: " << atoms.n_builds << "\n\n";
    }

    // --- 2) Timing: periodic LJ liquid, rho = 0.8, T = 1.0 -----------------------
    const int max_n = argc > 1 ? std::stoi(argv[1]) : 24;   // lattice side -> N = n^3
    const double a = std::cbrt(1.0 / 0.8);
    std::cout << std::setw(8) << "N" << std::setw(14) << "CPU nlist" << std::setw(14) << "CPU N^2"
#ifdef WITH_CUDA
              << std::setw(14) << "GPU nlist" << std::setw(14) << "GPU N^2"
#endif
              << "   [ms/step]\n";

    for (int n = 6; n <= max_n; n += 6) {
        const auto x = cubic_lattice(n, a);
        MDParams p;
        p.box = {n * a, n * a, n * a};
        const std::size_t N = x.size() / 3;
        const int steps = std::max(5, static_cast<int>(2e6 / N));
        std::cout << std::setw(8) << N << std::fixed << std::setprecision(4);

        for (bool nl : {true, false}) {
            p.use_nlist = nl;
            if (!nl && N > 10000) { std::cout << std::setw(14) << "-"; continue; }
            AtomsCPU cpu(x, p);
            cpu.run(2, true);   // warm-up
            std::cout << std::setw(14) << 1e3 * seconds_per_step(cpu, nl ? steps : std::max(2, steps / 20));
        }
#ifdef WITH_CUDA
        for (bool nl : {true, false}) {
            p.use_nlist = nl;
            AtomsGPU gpu(x, p);
            gpu.run(20, true);  // warm-up
            std::cout << std::setw(14) << 1e3 * seconds_per_step(gpu, nl ? 10 * steps : std::max(20, steps));
        }
#endif
        std::cout << std::defaultfloat << "\n";
    }
}
