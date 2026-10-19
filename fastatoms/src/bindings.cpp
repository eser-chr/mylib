#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "atoms.hpp"
#include "elements.hpp"
#include "extxyz.hpp"

// info is exposed as a mutable, reference-semantics mapping so that
// `atoms.info["energy"] = -1.0` modifies the C++ object in place.
PYBIND11_MAKE_OPAQUE(fastatoms::InfoMap)

namespace py = pybind11;
using namespace fastatoms;

namespace {

constexpr int kFlags = py::array::c_style | py::array::forcecast;
using DoubleArray = py::array_t<double, kFlags>;
using Int64Array = py::array_t<std::int64_t, kFlags>;
using BoolArray = py::array_t<bool, kFlags>;

// ------------------------------------------------------------------ conversions

std::string fspath(const py::object& path) {
    return py::module_::import("os").attr("fspath")(path).cast<std::string>();
}

py::module_ numpy() { return py::module_::import("numpy"); }

int checked_number(std::int64_t z) {
    if (z < 0 || z > kMaxZ) throw std::invalid_argument("atomic number " + std::to_string(z) + " out of range");
    return static_cast<int>(z);
}

int number_from_handle(py::handle obj) {
    if (py::isinstance<py::str>(obj)) return symbol_to_number(obj.cast<std::string>());
    if (py::isinstance<Atom>(obj)) return obj.cast<const Atom&>().number;
    return checked_number(obj.cast<std::int64_t>());
}

std::vector<int> numbers_from_object(const py::object& obj) {
    if (py::isinstance<py::str>(obj)) return parse_formula(obj.cast<std::string>());
    std::vector<int> out;
    for (py::handle item : obj) out.push_back(number_from_handle(item));
    return out;
}

std::vector<double> positions_from_object(const py::object& obj, std::size_t natoms) {
    DoubleArray arr = obj.cast<DoubleArray>();
    if (arr.size() == 0 && natoms == 0) return {};
    if (arr.ndim() != 2 || arr.shape(1) != 3) throw std::invalid_argument("positions must have shape (N, 3)");
    if (static_cast<std::size_t>(arr.shape(0)) != natoms) {
        throw std::invalid_argument("got positions for " + std::to_string(arr.shape(0)) +
                                    " atoms but the structure has " + std::to_string(natoms));
    }
    return std::vector<double>(arr.data(), arr.data() + arr.size());
}

Vec3 vec3_from_object(const py::object& obj) {
    DoubleArray arr = obj.cast<DoubleArray>();
    if (arr.size() != 3) throw std::invalid_argument("expected a vector with 3 components");
    const double* d = arr.data();
    return {d[0], d[1], d[2]};
}

Mat3 cell_from_object(const py::object& obj) {
    DoubleArray arr = obj.cast<DoubleArray>();
    const double* d = arr.data();
    Mat3 m{};
    if (arr.size() == 3) {
        m[0] = d[0];
        m[4] = d[1];
        m[8] = d[2];
    } else if (arr.size() == 9) {
        std::copy(d, d + 9, m.begin());
    } else {
        throw std::invalid_argument("cell must be 3 lengths or a 3x3 matrix");
    }
    return m;
}

std::array<bool, 3> pbc_from_object(const py::object& obj) {
    BoolArray arr = obj.cast<BoolArray>();
    const bool* d = arr.data();
    if (arr.size() == 1) return {d[0], d[0], d[0]};
    if (arr.size() == 3) return {d[0], d[1], d[2]};
    throw std::invalid_argument("pbc must be a bool or three bools");
}

template <class T>
py::array_t<T> make_array(const T* data, std::vector<py::ssize_t> shape) {
    py::array_t<T> out(shape);
    py::ssize_t total = 1;
    for (auto s : shape) total *= s;
    if (total > 0) std::memcpy(out.mutable_data(), data, sizeof(T) * static_cast<std::size_t>(total));
    return out;
}

py::array_t<double> positions_array(const Atoms& a) {
    return make_array(a.positions.data(), {static_cast<py::ssize_t>(a.size()), 3});
}

py::array_t<std::int64_t> numbers_array(const Atoms& a) {
    py::array_t<std::int64_t> out(static_cast<py::ssize_t>(a.size()));
    auto* d = out.mutable_data();
    for (std::size_t i = 0; i < a.size(); ++i) d[i] = a.numbers[i];
    return out;
}

py::array_t<double> mat3_array(const Mat3& m) { return make_array(m.data(), {3, 3}); }

py::array_t<bool> pbc_array(const Atoms& a) {
    py::array_t<bool> out(3);
    auto* d = out.mutable_data();
    for (int i = 0; i < 3; ++i) d[i] = a.pbc[static_cast<std::size_t>(i)];
    return out;
}

py::list symbols_list(const Atoms& a) {
    py::list out;
    for (int z : a.numbers) out.append(py::str(number_to_symbol(z)));
    return out;
}

void set_numbers_checked(Atoms& a, std::vector<int> numbers) {
    if (numbers.size() != a.size()) {
        throw std::invalid_argument("expected " + std::to_string(a.size()) + " symbols/numbers, got " +
                                    std::to_string(numbers.size()));
    }
    a.numbers = std::move(numbers);
}

// -------------------------------------------------------------- per-atom arrays

bool is_reserved(const std::string& name) {
    return name == "numbers" || name == "positions" || name == "species" || name == "pos";
}

py::object per_atom_to_python(const PerAtomArray& arr, std::size_t n) {
    const auto rows = static_cast<py::ssize_t>(n);
    std::vector<py::ssize_t> shape = arr.ncols == 1 ? std::vector<py::ssize_t>{rows}
                                                    : std::vector<py::ssize_t>{rows, arr.ncols};
    switch (arr.type) {
        case PropType::Real: return make_array(arr.real.data(), shape);
        case PropType::Int: return make_array(arr.integer.data(), shape);
        case PropType::Bool: {
            py::array_t<bool> out(shape);
            auto* d = out.mutable_data();
            for (std::size_t i = 0; i < arr.logical.size(); ++i) d[i] = arr.logical[i] != 0;
            return out;
        }
        case PropType::Str: {
            py::list out;
            for (std::size_t i = 0; i < n; ++i) {
                if (arr.ncols == 1) {
                    out.append(py::str(arr.str[i]));
                } else {
                    py::list row;
                    for (int c = 0; c < arr.ncols; ++c) {
                        row.append(py::str(arr.str[i * static_cast<std::size_t>(arr.ncols) + static_cast<std::size_t>(c)]));
                    }
                    out.append(row);
                }
            }
            return out;
        }
    }
    return py::none();
}

PerAtomArray per_atom_from_python(const py::object& values, std::size_t n) {
    py::array arr = numpy().attr("asarray")(values);
    if (arr.ndim() < 1 || arr.ndim() > 2) throw std::invalid_argument("per-atom arrays must be 1D or 2D");
    if (static_cast<std::size_t>(arr.shape(0)) != n) {
        throw std::invalid_argument("array has " + std::to_string(arr.shape(0)) + " rows but the structure has " +
                                    std::to_string(n) + " atoms");
    }
    PerAtomArray out;
    out.ncols = arr.ndim() == 2 ? static_cast<int>(arr.shape(1)) : 1;
    if (out.ncols < 1) throw std::invalid_argument("per-atom arrays need at least one column");

    const char kind = arr.dtype().kind();
    if (kind == 'f') {
        out.type = PropType::Real;
        auto a = arr.cast<DoubleArray>();
        out.real.assign(a.data(), a.data() + a.size());
    } else if (kind == 'i' || kind == 'u') {
        out.type = PropType::Int;
        auto a = arr.cast<Int64Array>();
        out.integer.assign(a.data(), a.data() + a.size());
    } else if (kind == 'b') {
        out.type = PropType::Bool;
        auto a = arr.cast<BoolArray>();
        out.logical.resize(static_cast<std::size_t>(a.size()));
        for (py::ssize_t i = 0; i < a.size(); ++i) out.logical[static_cast<std::size_t>(i)] = a.data()[i] ? 1 : 0;
    } else if (kind == 'U' || kind == 'S' || kind == 'O') {
        out.type = PropType::Str;
        py::object flat = arr.attr("ravel")().attr("tolist")();
        for (py::handle item : flat) {
            std::string s = py::isinstance<py::bytes>(item) ? item.cast<std::string>() : py::str(item).cast<std::string>();
            if (s.empty() || std::any_of(s.begin(), s.end(), [](char c) { return std::isspace(static_cast<unsigned char>(c)); })) {
                throw std::invalid_argument("string per-atom values must be non-empty and contain no whitespace");
            }
            out.str.push_back(std::move(s));
        }
    } else {
        throw std::invalid_argument("unsupported dtype for per-atom array");
    }
    return out;
}

// ------------------------------------------------------------------------- info

py::dict info_to_dict(const InfoMap& m) {
    py::dict d;
    for (const auto& kv : m) d[py::str(kv.first)] = py::cast(kv.second);
    return d;
}

InfoMap info_from_object(const py::object& obj) {
    if (py::isinstance<InfoMap>(obj)) return obj.cast<const InfoMap&>();
    InfoMap m;
    py::dict d(obj);
    for (auto item : d) m[item.first.cast<std::string>()] = item.second.cast<InfoValue>();
    return m;
}

// ---------------------------------------------------------------------- indexing

bool is_scalar_index(const py::object& key) {
    return PyIndex_Check(key.ptr()) && !py::isinstance<py::array>(key);
}

std::vector<std::int64_t> indices_from_object(const Atoms& a, const py::object& key) {
    const auto n = static_cast<py::ssize_t>(a.size());
    std::vector<std::int64_t> idx;
    if (py::isinstance<py::slice>(key)) {
        py::ssize_t start = 0, stop = 0, step = 0, length = 0;
        if (!key.cast<py::slice>().compute(n, &start, &stop, &step, &length)) throw py::error_already_set();
        idx.reserve(static_cast<std::size_t>(length));
        for (py::ssize_t k = 0; k < length; ++k) idx.push_back(start + k * step);
        return idx;
    }
    py::array arr = numpy().attr("asarray")(key);
    if (arr.dtype().kind() == 'b') {
        if (arr.ndim() != 1 || arr.shape(0) != n) throw py::index_error("boolean mask must have length len(atoms)");
        auto mask = arr.cast<BoolArray>();
        for (py::ssize_t i = 0; i < n; ++i) {
            if (mask.data()[i]) idx.push_back(i);
        }
    } else {
        auto ia = arr.cast<Int64Array>();
        idx.assign(ia.data(), ia.data() + ia.size());
    }
    return idx;
}

// ------------------------------------------------------------------------ misc

std::string chemical_formula(const Atoms& a, const std::string& mode) {
    std::vector<int> order;
    std::map<int, std::size_t> counts;
    for (int z : a.numbers) {
        if (counts[z]++ == 0) order.push_back(z);
    }
    if (mode == "hill") {
        const bool has_carbon = counts.count(6) > 0;
        auto rank = [&](int z) {
            if (has_carbon && z == 6) return 0;
            if (has_carbon && z == 1) return 1;
            return 2;
        };
        std::sort(order.begin(), order.end(), [&](int x, int y) {
            const int rx = rank(x), ry = rank(y);
            if (rx != ry) return rx < ry;
            return std::strcmp(number_to_symbol(x), number_to_symbol(y)) < 0;
        });
    } else if (mode != "count") {
        throw std::invalid_argument("mode must be 'hill' or 'count'");
    }
    std::string out;
    for (int z : order) {
        out += number_to_symbol(z);
        if (counts[z] > 1) out += std::to_string(counts[z]);
    }
    return out;
}

std::string atoms_repr(const Atoms& a) {
    std::ostringstream os;
    os.precision(6);
    os << "Atoms(symbols='" << chemical_formula(a, "hill") << "', pbc=[";
    for (std::size_t i = 0; i < 3; ++i) os << (i ? ", " : "") << (a.pbc[i] ? "True" : "False");
    os << "], cell=[";
    for (std::size_t r = 0; r < 3; ++r) {
        os << (r ? ", " : "") << "[" << a.cell[3 * r] << ", " << a.cell[3 * r + 1] << ", " << a.cell[3 * r + 2] << "]";
    }
    os << "])";
    return os.str();
}

Atoms make_atoms(const py::object& symbols, const py::object& positions, const py::object& numbers,
                 const py::object& cell, const py::object& pbc, const py::object& info) {
    if (!symbols.is_none() && !numbers.is_none()) {
        throw std::invalid_argument("give either symbols or numbers, not both");
    }
    Atoms a;
    if (!symbols.is_none()) {
        if (py::isinstance<Atoms>(symbols)) {
            a = symbols.cast<const Atoms&>();
        } else if (py::isinstance<py::str>(symbols)) {
            a.numbers = parse_formula(symbols.cast<std::string>());
        } else {
            bool all_atom_objects = true;
            std::vector<double> pos;
            for (py::handle item : symbols) {
                if (py::isinstance<Atom>(item)) {
                    const Atom& at = item.cast<const Atom&>();
                    a.numbers.push_back(at.number);
                    pos.insert(pos.end(), at.position.begin(), at.position.end());
                } else {
                    all_atom_objects = false;
                    a.numbers.push_back(number_from_handle(item));
                }
            }
            if (all_atom_objects) a.positions = std::move(pos);
        }
    } else if (!numbers.is_none()) {
        a.numbers = numbers_from_object(numbers);
    }

    if (!positions.is_none()) a.positions = positions_from_object(positions, a.size());
    if (a.positions.size() != 3 * a.size()) a.positions.assign(3 * a.size(), 0.0);
    if (!cell.is_none()) a.cell = cell_from_object(cell);
    if (!pbc.is_none()) a.pbc = pbc_from_object(pbc);
    if (!info.is_none()) a.info = info_from_object(info);
    return a;
}

// ------------------------------------------------------------------------- I/O

std::vector<std::size_t> selection_from_index(const py::object& index, std::size_t nframes, bool& single) {
    single = false;
    std::vector<std::size_t> sel;
    if (index.is_none()) {
        sel.resize(nframes);
        for (std::size_t i = 0; i < nframes; ++i) sel[i] = i;
        return sel;
    }
    if (py::isinstance<py::slice>(index)) {
        py::ssize_t start = 0, stop = 0, step = 0, length = 0;
        if (!index.cast<py::slice>().compute(static_cast<py::ssize_t>(nframes), &start, &stop, &step, &length)) {
            throw py::error_already_set();
        }
        sel.reserve(static_cast<std::size_t>(length));
        for (py::ssize_t k = 0; k < length; ++k) sel.push_back(static_cast<std::size_t>(start + k * step));
        return sel;
    }
    std::int64_t i = index.cast<std::int64_t>();
    const auto n = static_cast<std::int64_t>(nframes);
    if (i < 0) i += n;
    if (i < 0 || i >= n) {
        throw py::index_error("frame index out of range, the file contains " + std::to_string(nframes) + " frames");
    }
    sel.push_back(static_cast<std::size_t>(i));
    single = true;
    return sel;
}

py::object parse_buffer(const std::string& data, const py::object& index) {
    std::vector<FrameSpan> spans;
    {
        py::gil_scoped_release release;
        spans = scan_extxyz_frames(data);
    }
    bool single = false;
    const std::vector<std::size_t> sel = selection_from_index(index, spans.size(), single);
    std::vector<Atoms> frames;
    {
        py::gil_scoped_release release;
        frames = parse_extxyz_frames(data, spans, sel);
    }
    if (single) return py::cast(std::move(frames[0]));
    py::list out;
    for (auto& f : frames) out.append(py::cast(std::move(f)));
    return out;
}

std::vector<const Atoms*> frames_from_object(const py::object& images, std::vector<py::object>& keep_alive) {
    std::vector<const Atoms*> frames;
    auto add = [&](py::handle h) {
        if (!py::isinstance<Atoms>(h)) throw py::type_error("images must be fastatoms.Atoms objects");
        keep_alive.push_back(py::reinterpret_borrow<py::object>(h));
        frames.push_back(h.cast<const Atoms*>());
    };
    if (py::isinstance<Atoms>(images)) {
        add(images);
    } else {
        for (py::handle h : images) add(h);
    }
    return frames;
}

}  // namespace

// ============================================================================

PYBIND11_MODULE(_core, m) {
    m.doc() = "C++ core of fastatoms, an ASE-like Atoms container with fast extxyz I/O";
    m.attr("__version__") = "0.1.0";
#ifdef FASTATOMS_HAVE_OPENMP
    m.attr("has_openmp") = true;
#else
    m.attr("has_openmp") = false;
#endif
#ifdef FASTATOMS_HAVE_FAST_FLOAT
    m.attr("has_fast_float") = true;
#else
    m.attr("has_fast_float") = false;
#endif

    // ---------------------------------------------------------------- Atom
    py::class_<Atom>(m, "Atom", "A single atom (a copy, changing it does not modify the parent Atoms)")
        .def(py::init([](const py::object& symbol, const py::object& position) {
                 Atom at;
                 at.number = number_from_handle(symbol);
                 at.position = vec3_from_object(position);
                 return at;
             }),
             py::arg("symbol") = "X", py::arg("position") = py::make_tuple(0.0, 0.0, 0.0))
        .def_property(
            "symbol", [](const Atom& a) { return std::string(number_to_symbol(a.number)); },
            [](Atom& a, const py::object& s) { a.number = number_from_handle(s); })
        .def_property(
            "number", [](const Atom& a) { return a.number; },
            [](Atom& a, std::int64_t z) { a.number = checked_number(z); })
        .def_property(
            "position", [](const Atom& a) { return make_array(a.position.data(), {3}); },
            [](Atom& a, const py::object& p) { a.position = vec3_from_object(p); })
        .def_readonly("index", &Atom::index)
        .def("__repr__", [](const Atom& a) {
            std::ostringstream os;
            os.precision(8);
            os << "Atom('" << number_to_symbol(a.number) << "', [" << a.position[0] << ", " << a.position[1] << ", "
               << a.position[2] << "], index=" << a.index << ")";
            return os.str();
        });

    // ------------------------------------------------------------ InfoDict
    py::bind_map<InfoMap>(m, "InfoDict")
        .def(
            "get",
            [](const InfoMap& d, const std::string& key, const py::object& def) -> py::object {
                auto it = d.find(key);
                return it == d.end() ? def : py::cast(it->second);
            },
            py::arg("key"), py::arg("default") = py::none())
        .def(
            "pop",
            [](InfoMap& d, const std::string& key, const py::object& def) -> py::object {
                auto it = d.find(key);
                if (it == d.end()) {
                    if (def.is_none()) throw py::key_error(key);
                    return def;
                }
                py::object value = py::cast(it->second);
                d.erase(it);
                return value;
            },
            py::arg("key"), py::arg("default") = py::none())
        .def("update",
             [](InfoMap& d, const py::object& other) {
                 for (auto& kv : info_from_object(other)) d[kv.first] = kv.second;
             })
        .def("to_dict", [](const InfoMap& d) { return info_to_dict(d); })
        .def("__repr__", [](const InfoMap& d) { return py::repr(info_to_dict(d)).cast<std::string>(); });

    // --------------------------------------------------------------- Atoms
    py::class_<Atoms>(m, "Atoms", "ASE-like collection of atoms backed by contiguous C++ storage")
        .def(py::init(&make_atoms), py::arg("symbols") = py::none(), py::arg("positions") = py::none(),
             py::arg("numbers") = py::none(), py::arg("cell") = py::none(), py::arg("pbc") = py::none(),
             py::arg("info") = py::none())
        .def("__len__", &Atoms::size)
        .def("__repr__", &atoms_repr)
        .def("copy", [](const Atoms& a) { return Atoms(a); })
        .def("__copy__", [](const Atoms& a) { return Atoms(a); })
        .def("__deepcopy__", [](const Atoms& a, const py::object&) { return Atoms(a); }, py::arg("memo"))

        // ---- adding / removing atoms
        .def(
            "append",
            [](Atoms& a, const py::object& atom, const py::object& position) {
                if (py::isinstance<Atom>(atom)) {
                    const Atom& at = atom.cast<const Atom&>();
                    a.append(at.number, position.is_none() ? at.position : vec3_from_object(position));
                } else {
                    a.append(number_from_handle(atom),
                             position.is_none() ? Vec3{0.0, 0.0, 0.0} : vec3_from_object(position));
                }
            },
            py::arg("atom"), py::arg("position") = py::none(),
            "Append one atom, given as an Atom, a symbol or an atomic number")
        .def(
            "extend",
            [](Atoms& a, const py::object& other) {
                if (py::isinstance<Atoms>(other)) {
                    a.extend(other.cast<const Atoms&>());
                } else if (py::isinstance<Atom>(other)) {
                    const Atom& at = other.cast<const Atom&>();
                    a.append(at.number, at.position);
                } else {
                    for (py::handle item : other) {
                        const Atom& at = item.cast<const Atom&>();
                        a.append(at.number, at.position);
                    }
                }
            },
            py::arg("other"))
        .def(
            "__iadd__", [](Atoms& a, const Atoms& o) -> Atoms& {
                a.extend(o);
                return a;
            },
            py::return_value_policy::reference_internal)
        .def("__add__",
             [](const Atoms& a, const Atoms& o) {
                 Atoms out(a);
                 out.extend(o);
                 return out;
             })
        .def("__getitem__",
             [](const Atoms& a, const py::object& key) -> py::object {
                 if (is_scalar_index(key)) return py::cast(a.get_atom(key.cast<std::int64_t>()));
                 return py::cast(a.take(indices_from_object(a, key)));
             })
        .def("__setitem__",
             [](Atoms& a, std::int64_t i, const Atom& at) {
                 const std::size_t k = a.normalize_index(i);
                 a.numbers[k] = at.number;
                 std::copy(at.position.begin(), at.position.end(), a.positions.begin() + 3 * k);
             })
        .def("__delitem__",
             [](Atoms& a, const py::object& key) {
                 if (is_scalar_index(key)) {
                     a.remove({key.cast<std::int64_t>()});
                 } else {
                     a.remove(indices_from_object(a, key));
                 }
             })
        .def(
            "pop",
            [](Atoms& a, std::int64_t i) {
                Atom at = a.get_atom(i);
                a.remove({i});
                return at;
            },
            py::arg("i") = -1)

        // ---- positions
        .def("get_positions", &positions_array)
        .def(
            "set_positions",
            [](Atoms& a, const py::object& p) { a.positions = positions_from_object(p, a.size()); },
            py::arg("newpositions"))
        .def_property("positions", &positions_array,
                      [](Atoms& a, const py::object& p) { a.positions = positions_from_object(p, a.size()); })
        .def(
            "get_scaled_positions",
            [](const Atoms& a, bool wrap) {
                const auto frac = a.scaled_positions(wrap);
                return make_array(frac.data(), {static_cast<py::ssize_t>(a.size()), 3});
            },
            py::arg("wrap") = true)
        .def(
            "set_scaled_positions",
            [](Atoms& a, const py::object& p) { a.set_scaled_positions(positions_from_object(p, a.size())); },
            py::arg("scaled"))

        // ---- species
        .def("get_chemical_symbols", &symbols_list)
        .def(
            "set_chemical_symbols",
            [](Atoms& a, const py::object& s) { set_numbers_checked(a, numbers_from_object(s)); }, py::arg("symbols"))
        .def_property("symbols", &symbols_list,
                      [](Atoms& a, const py::object& s) { set_numbers_checked(a, numbers_from_object(s)); })
        .def("get_atomic_numbers", &numbers_array)
        .def(
            "set_atomic_numbers",
            [](Atoms& a, const py::object& z) { set_numbers_checked(a, numbers_from_object(z)); }, py::arg("numbers"))
        .def_property("numbers", &numbers_array,
                      [](Atoms& a, const py::object& z) { set_numbers_checked(a, numbers_from_object(z)); })
        .def("get_chemical_formula", &chemical_formula, py::arg("mode") = "hill")

        // ---- cell / pbc
        .def("get_cell", [](const Atoms& a) { return mat3_array(a.cell); })
        .def(
            "set_cell",
            [](Atoms& a, const py::object& c, bool scale_atoms) { a.set_cell(cell_from_object(c), scale_atoms); },
            py::arg("cell"), py::arg("scale_atoms") = false)
        .def_property(
            "cell", [](const Atoms& a) { return mat3_array(a.cell); },
            [](Atoms& a, const py::object& c) { a.cell = cell_from_object(c); })
        .def("get_pbc", &pbc_array)
        .def(
            "set_pbc", [](Atoms& a, const py::object& p) { a.pbc = pbc_from_object(p); }, py::arg("pbc"))
        .def_property("pbc", &pbc_array, [](Atoms& a, const py::object& p) { a.pbc = pbc_from_object(p); })
        .def("get_volume", &Atoms::volume)

        // ---- geometry
        .def(
            "translate", [](Atoms& a, const py::object& d) { a.translate(vec3_from_object(d)); },
            py::arg("displacement"))
        .def("wrap", &Atoms::wrap, py::arg("eps") = 1e-7)
        .def("get_distance", &Atoms::distance, py::arg("a0"), py::arg("a1"), py::arg("mic") = false)
        .def(
            "get_all_distances",
            [](const Atoms& a, bool mic) {
                std::vector<double> d;
                {
                    py::gil_scoped_release release;
                    d = a.all_distances(mic);
                }
                const auto n = static_cast<py::ssize_t>(a.size());
                return make_array(d.data(), {n, n});
            },
            py::arg("mic") = false)

        // ---- info and per-atom arrays
        .def_property(
            "info", [](Atoms& a) -> InfoMap& { return a.info; },
            [](Atoms& a, const py::object& d) { a.info = info_from_object(d); })
        .def(
            "get_array",
            [](const Atoms& a, const std::string& name) -> py::object {
                if (name == "numbers") return numbers_array(a);
                if (name == "positions") return positions_array(a);
                auto it = a.arrays.find(name);
                if (it == a.arrays.end()) throw py::key_error(name);
                return per_atom_to_python(it->second, a.size());
            },
            py::arg("name"))
        .def(
            "set_array",
            [](Atoms& a, const std::string& name, const py::object& values) {
                if (values.is_none()) {
                    a.arrays.erase(name);
                    return;
                }
                if (is_reserved(name)) {
                    throw std::invalid_argument("'" + name +
                                                "' is stored natively, use set_positions/set_atomic_numbers");
                }
                a.arrays[name] = per_atom_from_python(values, a.size());
            },
            py::arg("name"), py::arg("values"))
        .def(
            "has",
            [](const Atoms& a, const std::string& name) {
                return name == "numbers" || name == "positions" || a.arrays.count(name) > 0;
            },
            py::arg("name"))
        .def(
            "delete_array", [](Atoms& a, const std::string& name) {
                if (a.arrays.erase(name) == 0) throw py::key_error(name);
            },
            py::arg("name"))
        .def("array_names",
             [](const Atoms& a) {
                 std::vector<std::string> names;
                 for (const auto& kv : a.arrays) names.push_back(kv.first);
                 return names;
             })
        .def_property_readonly(
            "arrays",
            [](const Atoms& a) {
                py::dict d;
                d["numbers"] = numbers_array(a);
                d["positions"] = positions_array(a);
                for (const auto& kv : a.arrays) d[py::str(kv.first)] = per_atom_to_python(kv.second, a.size());
                return d;
            },
            "Snapshot (copies) of all per-atom arrays")

        // ---- I/O convenience
        .def(
            "write",
            [](const Atoms& a, const py::object& filename, bool append) {
                const std::string path = fspath(filename);
                std::vector<const Atoms*> frames{&a};
                py::gil_scoped_release release;
                write_extxyz(path, frames, append);
            },
            py::arg("filename"), py::arg("append") = false)

        // ---- pickling (keeps full precision, unlike going through text)
        .def(py::pickle(
            [](const Atoms& a) {
                py::dict arrays;
                for (const auto& kv : a.arrays) arrays[py::str(kv.first)] = per_atom_to_python(kv.second, a.size());
                return py::make_tuple(numbers_array(a), positions_array(a), mat3_array(a.cell), pbc_array(a),
                                      info_to_dict(a.info), arrays);
            },
            [](const py::tuple& t) {
                if (t.size() != 6) throw std::runtime_error("invalid pickle state for Atoms");
                Atoms a;
                a.numbers = numbers_from_object(t[0]);
                a.positions = positions_from_object(t[1], a.size());
                a.cell = cell_from_object(t[2]);
                a.pbc = pbc_from_object(t[3]);
                a.info = info_from_object(t[4]);
                for (auto item : t[5].cast<py::dict>()) {
                    a.arrays[item.first.cast<std::string>()] =
                        per_atom_from_python(py::reinterpret_borrow<py::object>(item.second), a.size());
                }
                return a;
            }));

    // ------------------------------------------------------------ extxyz
    m.def(
        "read_extxyz",
        [](const py::object& filename, const py::object& index) {
            const std::string path = fspath(filename);
            std::string data;
            {
                py::gil_scoped_release release;
                data = read_file(path);
            }
            return parse_buffer(data, index);
        },
        py::arg("filename"), py::arg("index") = -1,
        "Read an extxyz file. index is an int (returns Atoms), a slice or None (returns a list)");

    m.def(
        "parse_extxyz", [](const std::string& text, const py::object& index) { return parse_buffer(text, index); },
        py::arg("text"), py::arg("index") = py::none(), "Parse extxyz content from a string");

    m.def(
        "write_extxyz",
        [](const py::object& filename, const py::object& images, bool append) {
            const std::string path = fspath(filename);
            std::vector<py::object> keep_alive;
            const std::vector<const Atoms*> frames = frames_from_object(images, keep_alive);
            py::gil_scoped_release release;
            write_extxyz(path, frames, append);
        },
        py::arg("filename"), py::arg("images"), py::arg("append") = false);

    m.def(
        "format_extxyz",
        [](const py::object& images) {
            std::vector<py::object> keep_alive;
            const std::vector<const Atoms*> frames = frames_from_object(images, keep_alive);
            std::string out;
            {
                py::gil_scoped_release release;
                out = format_extxyz(frames);
            }
            return out;
        },
        py::arg("images"), "Format one Atoms or a sequence of Atoms as extxyz text");

    m.def(
        "count_extxyz_frames",
        [](const py::object& filename) {
            const std::string path = fspath(filename);
            py::gil_scoped_release release;
            return scan_extxyz_frames(read_file(path)).size();
        },
        py::arg("filename"));
}
