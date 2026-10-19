#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace fastatoms {

using Vec3 = std::array<double, 3>;
// Row-major 3x3 matrix. Row i is lattice vector i (same convention as ASE).
using Mat3 = std::array<double, 9>;

// Values stored in Atoms.info (per-frame metadata such as energy, stress, config_type).
using InfoValue = std::variant<bool, std::int64_t, double, std::string, std::vector<double>>;
using InfoMap = std::map<std::string, InfoValue>;

// Column types used by the extxyz "Properties" key.
enum class PropType : char { Real = 'R', Int = 'I', Bool = 'L', Str = 'S' };

// A per-atom property with ncols columns (e.g. forces is Real with ncols = 3).
// Only the vector that matches `type` is used. Data is row-major.
struct PerAtomArray {
    PropType type = PropType::Real;
    int ncols = 1;
    std::vector<double> real;
    std::vector<std::int64_t> integer;
    std::vector<std::uint8_t> logical;
    std::vector<std::string> str;

    std::size_t nrows() const;
    void resize_rows(std::size_t n);  // new rows are zero / false / empty
    PerAtomArray take(const std::vector<std::size_t>& rows) const;
    void append(const PerAtomArray& other);
};

struct Atom {
    int number = 0;
    Vec3 position{0.0, 0.0, 0.0};
    std::int64_t index = -1;  // index inside the parent Atoms, -1 if standalone
};

class Atoms {
public:
    std::vector<int> numbers;
    std::vector<double> positions;  // contiguous N x 3
    Mat3 cell{};
    std::array<bool, 3> pbc{false, false, false};
    InfoMap info;
    std::map<std::string, PerAtomArray> arrays;

    std::size_t size() const { return numbers.size(); }
    std::size_t normalize_index(std::int64_t i) const;

    void reserve(std::size_t n);
    void append(int number, const Vec3& position);
    void extend(const Atoms& other);
    Atom get_atom(std::int64_t i) const;
    Atoms take(const std::vector<std::int64_t>& indices) const;
    void remove(const std::vector<std::int64_t>& indices);

    std::vector<double> scaled_positions(bool wrap = false) const;
    void set_scaled_positions(const std::vector<double>& frac);
    void set_cell(const Mat3& new_cell, bool scale_atoms);
    double volume() const;
    void translate(const Vec3& d);
    void wrap(double eps = 1e-7);
    double distance(std::int64_t i, std::int64_t j, bool mic) const;
    std::vector<double> all_distances(bool mic) const;
};

Mat3 inverse(const Mat3& m);
double determinant(const Mat3& m);

}  // namespace fastatoms
