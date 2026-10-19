#pragma once
#include "common.hpp"

// Serial CPU engine: LJ + Velocity Verlet (NVE) / BAOAB Langevin,
// with a cell-list-built Verlet neighbour list (half list, Newton's 3rd law).
class AtomsCPU {
public:
    std::vector<double> pos;     // flat, 3N
    std::vector<double> vel;
    std::vector<double> forces;

    std::size_t N;
    MDParams p;
    double epot = 0.0;
    long n_builds = 0;           // number of neighbour-list builds

    std::mt19937_64 rng;
    std::normal_distribution<double> gauss{0.0, 1.0};

    AtomsCPU(std::vector<double> pos_, const MDParams& params)
        : pos(std::move(pos_)), forces(pos.size(), 0.0),
          N(pos.size() / 3), p(params), rng(params.seed)
    {
        if (pos.size() % 3 != 0 || N < 2) throw std::invalid_argument("need 3N positions, N >= 2");
        p.validate();
        periodic = p.periodic();
        if (periodic) for (int d = 0; d < 3; ++d) inv_box[d] = 1.0 / p.box[d];
        vel = maxwell_boltzmann(N, p.kT, p.mass, rng);
        if (p.use_nlist) build_nlist();
        compute_forces();
    }

    void set_velocities(const std::vector<double>& v) {
        if (v.size() != 3 * N) throw std::invalid_argument("velocities must have 3N entries");
        vel = v;
    }

    // ---------------------------------------------------------------- neighbour list
    // CSR half list: neighbours of i are nb_list[nb_start[i] .. nb_start[i+1]), all j > i.
    std::vector<int> nb_start, nb_list;
    std::vector<double> pos0;    // positions at last build
    std::vector<int> cell_head, cell_next;

    bool periodic = false;
    double inv_box[3] = {0.0, 0.0, 0.0};

    void min_image(double& dx, int d) const {
        if (periodic) dx -= p.box[d] * std::nearbyint(dx * inv_box[d]);
    }

    double dist2(std::size_t i, std::size_t j, double& dx, double& dy, double& dz) const {
        dx = pos[3*i]     - pos[3*j];     min_image(dx, 0);
        dy = pos[3*i + 1] - pos[3*j + 1]; min_image(dy, 1);
        dz = pos[3*i + 2] - pos[3*j + 2]; min_image(dz, 2);
        return dx*dx + dy*dy + dz*dz;
    }

    void build_nlist() {
        ++n_builds;
        // Wrap into the box. Positions stay unwrapped between builds, so pos - pos0 is the true displacement.
        if (p.periodic())
            for (std::size_t k = 0; k < 3*N; ++k)
                pos[k] -= p.box[k % 3] * std::floor(pos[k] / p.box[k % 3]);
        pos0 = pos;

        double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
        for (std::size_t i = 0; i < N; ++i)
            for (int d = 0; d < 3; ++d) {
                lo[d] = std::min(lo[d], pos[3*i + d]);
                hi[d] = std::max(hi[d], pos[3*i + d]);
            }
        const CellGrid g = make_grid(p, lo, hi, N);
        const double rl2 = p.rlist() * p.rlist();
        double dx, dy, dz;

        nb_start.assign(N + 1, 0);
        nb_list.clear();

        if (!g.ok) {   // small periodic box: all pairs
            for (std::size_t i = 0; i < N; ++i) {
                nb_start[i] = static_cast<int>(nb_list.size());
                for (std::size_t j = i + 1; j < N; ++j)
                    if (dist2(i, j, dx, dy, dz) < rl2) nb_list.push_back(static_cast<int>(j));
            }
            nb_start[N] = static_cast<int>(nb_list.size());
            return;
        }

        // Linked-cell list
        auto cell_coord = [&](std::size_t i, int d) {
            const int c = static_cast<int>(std::floor((pos[3*i + d] - g.lo[d]) * g.inv[d]));
            return std::clamp(c, 0, g.n[d] - 1);
        };
        cell_head.assign(g.ncells(), -1);
        cell_next.assign(N, -1);
        for (std::size_t i = 0; i < N; ++i) {
            const int c = (cell_coord(i, 2) * g.n[1] + cell_coord(i, 1)) * g.n[0] + cell_coord(i, 0);
            cell_next[i] = cell_head[c];
            cell_head[c] = static_cast<int>(i);
        }

        for (std::size_t i = 0; i < N; ++i) {
            nb_start[i] = static_cast<int>(nb_list.size());
            const int ci[3] = {cell_coord(i, 0), cell_coord(i, 1), cell_coord(i, 2)};
            for (int oz = -1; oz <= 1; ++oz)
            for (int oy = -1; oy <= 1; ++oy)
            for (int ox = -1; ox <= 1; ++ox) {
                int c[3] = {ci[0] + ox, ci[1] + oy, ci[2] + oz};
                bool skip = false;
                for (int d = 0; d < 3; ++d) {
                    if (p.periodic())                  c[d] = (c[d] + g.n[d]) % g.n[d];
                    else if (c[d] < 0 || c[d] >= g.n[d]) skip = true;
                }
                if (skip) continue;
                for (int j = cell_head[(c[2] * g.n[1] + c[1]) * g.n[0] + c[0]]; j != -1; j = cell_next[j])
                    if (static_cast<std::size_t>(j) > i && dist2(i, j, dx, dy, dz) < rl2)
                        nb_list.push_back(j);
            }
        }
        nb_start[N] = static_cast<int>(nb_list.size());
    }

    void maybe_rebuild() {
        if (!p.use_nlist) return;
        const double lim = 0.25 * p.skin * p.skin;   // (skin/2)^2
        for (std::size_t i = 0; i < N; ++i) {
            const double dx = pos[3*i] - pos0[3*i], dy = pos[3*i+1] - pos0[3*i+1], dz = pos[3*i+2] - pos0[3*i+2];
            if (dx*dx + dy*dy + dz*dz > lim) { build_nlist(); return; }
        }
    }

    // ---------------------------------------------------------------- forces
    void pair(std::size_t i, std::size_t j, double rc2, double s2) {
        double dx, dy, dz;
        const double r2 = dist2(i, j, dx, dy, dz);
        if (r2 > rc2) return;

        const double sr2  = s2 / r2;
        const double sr6  = sr2 * sr2 * sr2;
        const double sr12 = sr6 * sr6;

        epot += 4.0 * p.epsilon * (sr12 - sr6);
        const double f = 24.0 * p.epsilon * (2.0 * sr12 - sr6) / r2;  // F_ij = f * r_ij

        forces[3*i]     += f * dx;  forces[3*j]     -= f * dx;
        forces[3*i + 1] += f * dy;  forces[3*j + 1] -= f * dy;
        forces[3*i + 2] += f * dz;  forces[3*j + 2] -= f * dz;
    }

    void compute_forces() {
        std::fill(forces.begin(), forces.end(), 0.0);
        epot = 0.0;
        const double rc2 = p.rcut * p.rcut;
        const double s2  = p.sigma * p.sigma;

        if (p.use_nlist) {
            for (std::size_t i = 0; i < N; ++i)
                for (int k = nb_start[i]; k < nb_start[i + 1]; ++k)
                    pair(i, nb_list[k], rc2, s2);
        } else {
            for (std::size_t i = 0; i < N; ++i)
                for (std::size_t j = i + 1; j < N; ++j)
                    pair(i, j, rc2, s2);
        }
    }

    // ---------------------------------------------------------------- integrators
    // Velocity Verlet (NVE)
    void verlet_step() {
        const double h = 0.5 * p.dt / p.mass;
        for (std::size_t k = 0; k < 3*N; ++k) {
            vel[k] += h * forces[k];
            pos[k] += p.dt * vel[k];
        }
        maybe_rebuild();
        compute_forces();
        for (std::size_t k = 0; k < 3*N; ++k) vel[k] += h * forces[k];
    }

    // Langevin, BAOAB splitting (Leimkuhler & Matthews)
    void langevin_step() {
        const double h  = 0.5 * p.dt / p.mass;
        const double c1 = std::exp(-p.gamma * p.dt);
        const double c2 = std::sqrt((1.0 - c1 * c1) * p.kT / p.mass);

        for (std::size_t k = 0; k < 3*N; ++k) {
            vel[k] += h * forces[k];                 // B
            pos[k] += 0.5 * p.dt * vel[k];           // A
            vel[k]  = c1 * vel[k] + c2 * gauss(rng); // O
            pos[k] += 0.5 * p.dt * vel[k];           // A
        }
        maybe_rebuild();
        compute_forces();
        for (std::size_t k = 0; k < 3*N; ++k) vel[k] += h * forces[k];  // B
    }

    void run(int nsteps, bool langevin) {
        for (int s = 0; s < nsteps; ++s) langevin ? langevin_step() : verlet_step();
    }

    double ekin() const {
        double e = 0.0;
        for (double v : vel) e += v * v;
        return 0.5 * p.mass * e;
    }

    double temperature() const { return 2.0 * ekin() / (3.0 * N); }  // k_B = 1
};
