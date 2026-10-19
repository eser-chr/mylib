#pragma once

#include <string_view>
#include <vector>

namespace fastatoms {

constexpr int kMaxZ = 118;

// "Cu" -> 29. Index 0 is the dummy element "X" (as in ASE).
// Throws std::invalid_argument for unknown symbols.
int symbol_to_number(std::string_view symbol);

// 29 -> "Cu". Throws std::out_of_range for invalid numbers.
const char* number_to_symbol(int z);

// Simple formulas without parentheses, e.g. "Cu4Ga2O" -> [29,29,29,29,31,31,8].
std::vector<int> parse_formula(std::string_view formula);

}  // namespace fastatoms
