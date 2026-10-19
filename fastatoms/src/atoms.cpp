#include "atoms.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace fastatoms {
namespace {

template <class T>
std::vector<T> take_rows(const std::vector<T>& v, const std::vector<std::size_t>& rows, int ncols) {
    std::vector<T> out;
    out.reserve(rows.size() * static_cast<std::size_t>(ncols));
    for (std::size_t r : rows) {
        const std::size_t base = r * static_cast<std::size_t>(ncols);
        for (int c = 0; c < ncols; ++c) out.push_back(v[base + static_cast<std::size_t>(c)]);
    }
    return out;
}

template <class T>
void append_vector(std::vector<T>& dst, const std::vector<T>& src) {
    dst.insert(dst.end(), src.begin(), src.end());
}

Vec3 minimum_image(Vec3 d, const Mat3& cell, const Mat3& inv, const std::array<bool, 3>& pbc) {
    Vec3 f{};
    for (int j = 0; j < 3; ++j) f[j] = d[0] * inv[j] + d[1] * inv[3 + j] + d[2] * inv[6 + j];
    for (int c = 0; c < 3; ++c) {
        if (pbc[c]) f[c] -= std::round(f[c]);
    }
    for (int j = 0; j < 3; ++j) d[j] = f[0] * cell[j] + f[1] * cell[3 + j] + f[2] * cell[6 + j];
    return d;
}

}  // namespace

// ---------------------------------------------------------------- PerAtomArray

std::size_t PerAtomArray::nrows() const {
    std::size_t n = 0;
    switch (type) {
        case PropType::Real: n = real.size(); break;
        case PropType::Int: n = integer.size(); break;
        case PropType::Bool: n = logical.size(); break;
        case PropType::Str: n = str.size(); break;
    }
    return ncols > 0 ? n / static_cast<std::size_t>(ncols) : 0;
}

void PerAtomArray::resize_rows(std::size_t n) {
    const std::size_t m = n * static_cast<std::size_t>(ncols);
    switch (type) {
        case PropType::Real: real.resize(m, 0.0); break;
        case PropType::Int: integer.resize(m, 0); break;
        case PropType::Bool: logical.resize(m, 0); break;
        case PropType::Str: str.resize(m); break;
    }
}

PerAtomArray PerAtomArray::take(const std::vector<std::size_t>& rows) const {
    PerAtomArray out;
    out.type = type;
    out.ncols = ncols;
    switch (type) {
        case PropType::Real: out.real = take_rows(real, rows, ncols); break;
        case PropType::Int: out.integer = take_rows(integer, rows, ncols); break;
        case PropType::Bool: out.logical = take_rows(logical, rows, ncols); break;
        case PropType::Str: out.str = take_rows(str, rows, ncols); break;
    }
    return out;
}

void PerAtomArray::append(const PerAtomArray& other) {
    if (other.type != type || other.ncols != ncols) {
        throw std::invalid_argument("cannot append per-atom arrays with different type or shape");
    }
    switch (type) {
        case PropType::Real: append_vector(real, other.real); break;
        case PropType::Int: append_vector(integer, other.integer); break;
        case PropType::Bool: append_vector(logical, other.logical); break;
        case PropType::Str: append_vector(str, other.str); break;
    }
}

// ----------------------------------------------------------------------- Atoms

std::size_t Atoms::normalize_index(std::int64_t i) const {
    const auto n = static_cast<std::int64_t>(size());
    if (i < 0) i += n;
    if (i < 0 || i >= n) throw std::out_of_range("atom index out of range");
    return static_cast<std::size_t>(i);
}

void Atoms::reserve(std::size_t n) {
    numbers.reserve(n);
    positions.reserve(3 * n);
}

void Atoms::append(int number, const Vec3& position) {
    numbers.push_back(number);
    positions.insert(positions.end(), position.begin(), position.end());
    for (auto& kv : arrays) kv.second.resize_rows(numbers.size());
}

void Atoms::extend(const Atoms& other) {
    if (&other == this) {
        const Atoms copy(other);
        extend(copy);
        return;
    }
    for (const auto& kv : other.arrays) {
        auto it = arrays.find(kv.first);
        if (it != arrays.end() && (it->second.type != kv.second.type || it->second.ncols != kv.second.ncols)) {
            throw std::invalid_argument("cannot extend, per-atom array '" + kv.first +
                                        "' has a different type or shape");
        }
    }

    const std::size_t n0 = size();
    const std::size_t n_total = n0 + other.size();
    numbers.insert(numbers.end(), other.numbers.begin(), other.numbers.end());
    positions.insert(positions.end(), other.positions.begin(), other.positions.end());

    for (auto& kv : arrays) {
        auto it = other.arrays.find(kv.first);
        if (it != other.arrays.end()) {
            kv.second.append(it->second);
        } else {
            kv.second.resize_rows(n_total);
        }
    }
    for (const auto& kv : other.arrays) {
        if (arrays.count(kv.first)) continue;
        PerAtomArray fresh;
        fresh.type = kv.second.type;
        fresh.ncols = kv.second.ncols;
        fresh.resize_rows(n0);
        fresh.append(kv.second);
        arrays.emplace(kv.first, std::move(fresh));
    }
}

Atom Atoms::get_atom(std::int64_t i) const {
    const std::size_t k = normalize_index(i);
    Atom a;
    a.number = numbers[k];
    a.position = {positions[3 * k], positions[3 * k + 1], positions[3 * k + 2]};
    a.index = static_cast<std::int64_t>(k);
    return a;
}

Atoms Atoms::take(const std::vector<std::int64_t>& indices) const {
    std::vector<std::size_t> rows;
    rows.reserve(indices.size());
    for (std::int64_t i : indices) rows.push_back(normalize_index(i));

    Atoms out;
    out.cell = cell;
    out.pbc = pbc;
    out.info = info;
    out.numbers.reserve(rows.size());
    out.positions.reserve(3 * rows.size());
    for (std::size_t r : rows) {
        out.numbers.push_back(numbers[r]);
        out.positions.push_back(positions[3 * r]);
        out.positions.push_back(positions[3 * r + 1]);
        out.positions.push_back(positions[3 * r + 2]);
    }
    for (const auto& kv : arrays) out.arrays.emplace(kv.first, kv.second.take(rows));
    return out;
}

void Atoms::remove(const std::vector<std::int64_t>& indices) {
    std::vector<char> drop(size(), 0);
    for (std::int64_t i : indices) drop[normalize_index(i)] = 1;
    std::vector<std::int64_t> keep;
    keep.reserve(size());
    for (std::size_t k = 0; k < size(); ++k) {
        if (!drop[k]) keep.push_back(static_cast<std::int64_t>(k));
    }
    *this = take(keep);
}

std::vector<double> Atoms::scaled_positions(bool wrap_pbc) const {
    const Mat3 inv = inverse(cell);
    const std::size_t n = size();
    std::vector<double> frac(3 * n);
    for (std::size_t a = 0; a < n; ++a) {
        const double* p = &positions[3 * a];
        for (int j = 0; j < 3; ++j) {
            double f = p[0] * inv[j] + p[1] * inv[3 + j] + p[2] * inv[6 + j];
            if (wrap_pbc && pbc[j]) {
                f -= std::floor(f);
                if (f >= 1.0) f -= 1.0;
            }
            frac[3 * a + static_cast<std::size_t>(j)] = f;
        }
    }
    return frac;
}

void Atoms::set_scaled_positions(const std::vector<double>& frac) {
    if (frac.size() != positions.size()) {
        throw std::invalid_argument("scaled positions have the wrong length");
    }
    const std::size_t n = size();
    for (std::size_t a = 0; a < n; ++a) {
        const double* f = &frac[3 * a];
        for (int j = 0; j < 3; ++j) {
            positions[3 * a + static_cast<std::size_t>(j)] = f[0] * cell[j] + f[1] * cell[3 + j] + f[2] * cell[6 + j];
        }
    }
}

void Atoms::set_cell(const Mat3& new_cell, bool scale_atoms) {
    if (scale_atoms) {
        const std::vector<double> frac = scaled_positions(false);
        cell = new_cell;
        set_scaled_positions(frac);
    } else {
        cell = new_cell;
    }
}

double Atoms::volume() const { return std::abs(determinant(cell)); }

void Atoms::translate(const Vec3& d) {
    const std::size_t n = size();
    for (std::size_t a = 0; a < n; ++a) {
        positions[3 * a] += d[0];
        positions[3 * a + 1] += d[1];
        positions[3 * a + 2] += d[2];
    }
}

void Atoms::wrap(double eps) {
    if (!(pbc[0] || pbc[1] || pbc[2])) return;
    std::vector<double> frac = scaled_positions(false);
    const std::size_t n = size();
    for (std::size_t a = 0; a < n; ++a) {
        for (int c = 0; c < 3; ++c) {
            if (!pbc[c]) continue;
            double& f = frac[3 * a + static_cast<std::size_t>(c)];
            f += eps;
            f -= std::floor(f);
            f -= eps;
        }
    }
    set_scaled_positions(frac);
}

double Atoms::distance(std::int64_t i, std::int64_t j, bool mic) const {
    const std::size_t a = normalize_index(i);
    const std::size_t b = normalize_index(j);
    Vec3 d{positions[3 * b] - positions[3 * a], positions[3 * b + 1] - positions[3 * a + 1],
           positions[3 * b + 2] - positions[3 * a + 2]};
    if (mic && (pbc[0] || pbc[1] || pbc[2])) d = minimum_image(d, cell, inverse(cell), pbc);
    return std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
}

std::vector<double> Atoms::all_distances(bool mic) const {
    const std::size_t n = size();
    std::vector<double> out(n * n, 0.0);
    const bool use_mic = mic && (pbc[0] || pbc[1] || pbc[2]);
    Mat3 inv{};
    if (use_mic) inv = inverse(cell);
    const auto sn = static_cast<std::ptrdiff_t>(n);

#ifdef FASTATOMS_HAVE_OPENMP
    #pragma omp parallel for schedule(static) if (n > 256)
#endif
    for (std::ptrdiff_t ii = 0; ii < sn; ++ii) {
        const auto i = static_cast<std::size_t>(ii);
        for (std::size_t j = i + 1; j < n; ++j) {
            Vec3 d{positions[3 * j] - positions[3 * i], positions[3 * j + 1] - positions[3 * i + 1],
                   positions[3 * j + 2] - positions[3 * i + 2]};
            if (use_mic) d = minimum_image(d, cell, inv, pbc);
            const double r = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
            out[i * n + j] = r;
            out[j * n + i] = r;
        }
    }
    return out;
}

// --------------------------------------------------------------- linear algebra

double determinant(const Mat3& m) {
    return m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) +
           m[2] * (m[3] * m[7] - m[4] * m[6]);
}

Mat3 inverse(const Mat3& m) {
    const double a = m[0], b = m[1], c = m[2];
    const double d = m[3], e = m[4], f = m[5];
    const double g = m[6], h = m[7], i = m[8];
    const double A = e * i - f * h;
    const double B = -(d * i - f * g);
    const double C = d * h - e * g;
    const double det = a * A + b * B + c * C;
    if (std::abs(det) < 1e-300) {
        throw std::domain_error("cell matrix is singular (did you set a cell?)");
    }
    return Mat3{A / det, -(b * i - c * h) / det, (b * f - c * e) / det,
                B / det, (a * i - c * g) / det,  -(a * f - c * d) / det,
                C / det, -(a * h - b * g) / det, (a * e - b * d) / det};
}

}  // namespace fastatoms
