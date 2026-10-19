#include "elements.hpp"

#include <array>
#include <cctype>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace fastatoms {
namespace {

constexpr std::array<const char*, kMaxZ + 1> kSymbols = {
    "X",  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
    "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca",
    "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
};

const std::unordered_map<std::string_view, int>& symbol_table() {
    static const std::unordered_map<std::string_view, int> table = [] {
        std::unordered_map<std::string_view, int> t;
        t.reserve(kSymbols.size() * 2);
        for (int z = 0; z <= kMaxZ; ++z) t.emplace(kSymbols[z], z);
        return t;
    }();
    return table;
}

}  // namespace

int symbol_to_number(std::string_view symbol) {
    const auto& table = symbol_table();
    auto it = table.find(symbol);
    if (it == table.end()) {
        throw std::invalid_argument("unknown chemical symbol '" + std::string(symbol) + "'");
    }
    return it->second;
}

const char* number_to_symbol(int z) {
    if (z < 0 || z > kMaxZ) {
        throw std::out_of_range("invalid atomic number " + std::to_string(z));
    }
    return kSymbols[static_cast<std::size_t>(z)];
}

std::vector<int> parse_formula(std::string_view formula) {
    std::vector<int> out;
    std::size_t i = 0;
    while (i < formula.size()) {
        if (!std::isupper(static_cast<unsigned char>(formula[i]))) {
            throw std::invalid_argument("cannot parse chemical formula '" + std::string(formula) + "'");
        }
        std::size_t j = i + 1;
        while (j < formula.size() && std::islower(static_cast<unsigned char>(formula[j]))) ++j;
        const int z = symbol_to_number(formula.substr(i, j - i));

        std::size_t k = j;
        std::size_t count = 0;
        while (k < formula.size() && std::isdigit(static_cast<unsigned char>(formula[k]))) {
            count = count * 10 + static_cast<std::size_t>(formula[k] - '0');
            ++k;
        }
        if (k == j) count = 1;
        out.insert(out.end(), count, z);
        i = k;
    }
    return out;
}

}  // namespace fastatoms
