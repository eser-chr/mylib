#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "atoms_cpu.hpp"
#ifdef WITH_CUDA
#include "atoms_gpu.cuh"
#endif

namespace py = pybind11;
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

static std::vector<double> to_flat(const Array& a) {
    if (a.ndim() != 2 || a.shape(1) != 3) throw std::invalid_argument("expected an (N, 3) array");
    return std::vector<double>(a.data(), a.data() + a.size());
}

static Array to_numpy(const std::vector<double>& v) {
    Array out({static_cast<py::ssize_t>(v.size() / 3), static_cast<py::ssize_t>(3)});
    std::copy(v.begin(), v.end(), out.mutable_data());
    return out;
}

static MDParams make_params(std::array<double, 3> box, double kT, double mass, double dt, double gamma,
                            double epsilon, double sigma, double rcut, double skin,
                            unsigned long long seed, bool use_nlist) {
    MDParams p;
    p.box = box; p.kT = kT; p.mass = mass; p.dt = dt; p.gamma = gamma;
    p.epsilon = epsilon; p.sigma = sigma; p.rcut = rcut; p.skin = skin;
    p.seed = seed; p.use_nlist = use_nlist;
    return p;
}

// Constructor keyword arguments, identical for both engines.
#define MD_INIT_ARGS                                                                              \
    py::arg("positions"), py::arg("box") = std::array<double, 3>{0.0, 0.0, 0.0},                  \
        py::arg("kT") = 1.0, py::arg("mass") = 1.0, py::arg("dt") = 0.005, py::arg("gamma") = 1.0, \
        py::arg("epsilon") = 1.0, py::arg("sigma") = 1.0, py::arg("rcut") = 2.5,                   \
        py::arg("skin") = 0.3, py::arg("seed") = 42, py::arg("use_nlist") = true

template <class Engine>
static Engine* construct(const Array& pos, std::array<double, 3> box, double kT, double mass, double dt,
                         double gamma, double epsilon, double sigma, double rcut, double skin,
                         unsigned long long seed, bool use_nlist) {
    return new Engine(to_flat(pos), make_params(box, kT, mass, dt, gamma, epsilon, sigma, rcut, skin, seed, use_nlist));
}

PYBIND11_MODULE(_mymd, m) {
    m.doc() = "Lennard-Jones MD engine: serial CPU and CUDA GPU implementations";

#ifdef WITH_CUDA
    m.attr("has_cuda") = true;
#else
    m.attr("has_cuda") = false;
#endif

    // Long-running calls release the GIL.
    py::class_<AtomsCPU>(m, "AtomsCPU")
        .def(py::init(&construct<AtomsCPU>), MD_INIT_ARGS)
        .def("verlet_step", &AtomsCPU::verlet_step, py::call_guard<py::gil_scoped_release>())
        .def("langevin_step", &AtomsCPU::langevin_step, py::call_guard<py::gil_scoped_release>())
        .def("run", &AtomsCPU::run, py::arg("nsteps"), py::arg("langevin") = false,
             py::call_guard<py::gil_scoped_release>())
        .def_property("positions", [](const AtomsCPU& a) { return to_numpy(a.pos); },
                      [](AtomsCPU& a, const Array& x) {
                          auto v = to_flat(x);
                          if (v.size() != 3 * a.N) throw std::invalid_argument("wrong number of atoms");
                          a.pos = std::move(v);
                          if (a.p.use_nlist) a.build_nlist();
                          a.compute_forces();
                      })
        .def_property("velocities", [](const AtomsCPU& a) { return to_numpy(a.vel); },
                      [](AtomsCPU& a, const Array& v) { a.set_velocities(to_flat(v)); })
        .def_property_readonly("forces", [](const AtomsCPU& a) { return to_numpy(a.forces); })
        .def_property_readonly("epot", [](const AtomsCPU& a) { return a.epot; })
        .def_property_readonly("ekin", &AtomsCPU::ekin)
        .def_property_readonly("temperature", &AtomsCPU::temperature)
        .def_property_readonly("n_builds", [](const AtomsCPU& a) { return a.n_builds; })
        .def_property_readonly("N", [](const AtomsCPU& a) { return a.N; })
        .def("synchronize", [](const AtomsCPU&) {});   // no-op, for API symmetry

#ifdef WITH_CUDA
    py::class_<AtomsGPU>(m, "AtomsGPU")
        .def(py::init(&construct<AtomsGPU>), MD_INIT_ARGS)
        .def("verlet_step", &AtomsGPU::verlet_step, py::call_guard<py::gil_scoped_release>())
        .def("langevin_step", &AtomsGPU::langevin_step, py::call_guard<py::gil_scoped_release>())
        .def("run", &AtomsGPU::run, py::arg("nsteps"), py::arg("langevin") = false,
             py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("positions", [](const AtomsGPU& a) { return to_numpy(a.get_positions()); })
        .def_property("velocities", [](const AtomsGPU& a) { return to_numpy(a.get_velocities()); },
                      [](AtomsGPU& a, const Array& v) { a.set_velocities(to_flat(v)); })
        .def_property_readonly("forces", [](const AtomsGPU& a) { return to_numpy(a.get_forces()); })
        .def_property_readonly("epot", &AtomsGPU::epot)
        .def_property_readonly("ekin", &AtomsGPU::ekin)
        .def_property_readonly("temperature", &AtomsGPU::temperature)
        .def_property_readonly("n_builds", [](const AtomsGPU& a) { return a.n_builds; })
        .def_property_readonly("max_neighbors", [](const AtomsGPU& a) { return a.max_neighbors; })
        .def_property_readonly("N", [](const AtomsGPU& a) { return a.N; })
        .def("synchronize", &AtomsGPU::synchronize, py::call_guard<py::gil_scoped_release>());
#endif
}
