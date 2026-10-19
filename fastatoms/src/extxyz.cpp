#include "extxyz.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <system_error>
#include <utility>
#include <variant>

#include "elements.hpp"

#ifdef FASTATOMS_HAVE_FAST_FLOAT
#include <fast_float/fast_float.h>
#endif

namespace fastatoms {
namespace {

// ------------------------------------------------------------------ low level

inline bool is_space(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\f' || c == '\v'; }

inline const char* find_eol(const char* p, const char* e) {
    const void* hit = std::memchr(p, '\n', static_cast<std::size_t>(e - p));
    return hit ? static_cast<const char*>(hit) : e;
}

inline const char* next_line(const char* eol, const char* e) { return eol < e ? eol + 1 : e; }

inline bool next_token(const char*& q, const char* e, std::string_view& tok) {
    while (q < e && is_space(*q)) ++q;
    if (q >= e) return false;
    const char* b = q;
    while (q < e && !is_space(*q)) ++q;
    tok = std::string_view(b, static_cast<std::size_t>(q - b));
    return true;
}

std::string_view trim(std::string_view s) {
    std::size_t b = 0, e = s.size();
    while (b < e && is_space(s[b])) ++b;
    while (e > b && is_space(s[e - 1])) --e;
    return s.substr(b, e - b);
}

bool iequals(std::string_view a, std::string_view b) {
    // Checks lowercase equivalence
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const char x = (a[i] >= 'A' && a[i] <= 'Z') ? char(a[i] - 'A' + 'a') : a[i];
        const char y = (b[i] >= 'A' && b[i] <= 'Z') ? char(b[i] - 'A' + 'a') : b[i];
        if (x != y) return false;
    }
    return true;
}

bool parse_int(std::string_view s, std::int64_t& out) {
    if (s.empty()) return false;
    std::size_t i = 0;
    bool neg = false;
    if (s[0] == '+' || s[0] == '-') {
        neg = s[0] == '-';
        i = 1;
    }
    if (i >= s.size()) return false;
    std::int64_t v = 0;
    for (; i < s.size(); ++i) {
        const char c = s[i];
        if (c < '0' || c > '9') return false;
        v = v * 10 + (c - '0');
    }
    out = neg ? -v : v;
    return true;
}

bool parse_double(std::string_view s, double& out) {
    if (s.empty()) return false;
    const char* b = s.data();
    const char* e = b + s.size();
    if (*b == '+') ++b;
    if (b == e) return false;
#ifdef FASTATOMS_HAVE_FAST_FLOAT
    auto r = fast_float::from_chars(b, e, out);
    return r.ec == std::errc() && r.ptr == e;
#else
    char buf[128];
    const auto n = static_cast<std::size_t>(e - b);
    if (n >= sizeof(buf)) return false;
    std::memcpy(buf, b, n);
    buf[n] = '\0';
    char* endp = nullptr;
    out = std::strtod(buf, &endp);
    return endp == buf + n;
#endif
}

bool parse_bool(std::string_view s, bool& out) {
    if (s == "T" || s == "True" || s == "true" || s == "TRUE") {
        out = true;
        return true;
    }
    if (s == "F" || s == "False" || s == "false" || s == "FALSE") {
        out = false;
        return true;
    }
    return false;
}

// ------------------------------------------------------------- comment line

struct KeyValue {
    std::string key;
    std::string value;
    bool has_value = false;
};

std::vector<KeyValue> parse_comment(const char* p, const char* e) {
    std::vector<KeyValue> out;
    while (true) {
        while (p < e && is_space(*p)) ++p;
        if (p >= e) break;

        const char* kb = p;
        while (p < e && !is_space(*p) && *p != '=') ++p;
        KeyValue kv;
        kv.key.assign(kb, p);

        const char* after_key = p;
        while (p < e && is_space(*p)) ++p;
        if (p < e && *p == '=') {
            ++p;
            while (p < e && is_space(*p)) ++p;
            kv.has_value = true;
            if (p < e && (*p == '"' || *p == '\'')) {
                const char quote = *p++;
                while (p < e && *p != quote) {
                    if (*p == '\\' && p + 1 < e) {
                        ++p;
                        kv.value.push_back(*p == 'n' ? '\n' : *p);
                        ++p;
                    } else {
                        kv.value.push_back(*p++);
                    }
                }
                if (p < e) ++p;  // closing quote
            } else {
                const char* vb = p;
                while (p < e && !is_space(*p)) ++p;
                kv.value.assign(vb, p);
            }
        } else {
            p = after_key;  // bare key means True
        }
        if (!kv.key.empty()) out.push_back(std::move(kv));
    }
    return out;
}

std::vector<std::string_view> tokenize(const std::string& s) {
    std::vector<std::string_view> toks;
    const char* q = s.data();
    const char* e = q + s.size();
    std::string_view tok;
    while (next_token(q, e, tok)) toks.push_back(tok);
    return toks;
}

InfoValue infer_info_value(const KeyValue& kv) {
    if (!kv.has_value) return InfoValue(true);
    const auto toks = tokenize(kv.value);
    if (toks.size() == 1) {
        bool b;
        std::int64_t i;
        double d;
        if (parse_bool(toks[0], b)) return InfoValue(b);
        if (parse_int(toks[0], i)) return InfoValue(i);
        if (parse_double(toks[0], d)) return InfoValue(d);
        return InfoValue(kv.value);
    }
    if (toks.size() > 1) {
        std::vector<double> vals;
        vals.reserve(toks.size());
        for (auto tok : toks) {
            double d;
            if (!parse_double(tok, d)) return InfoValue(kv.value);
            vals.push_back(d);
        }
        return InfoValue(std::move(vals));
    }
    return InfoValue(kv.value);
}

struct ColumnSpec {
    std::string name;
    PropType type;
    int ncols;
};

std::vector<ColumnSpec> parse_properties(const std::string& s) {
    std::vector<std::string_view> parts;
    std::size_t start = 0;
    while (true) {
        const std::size_t pos = s.find(':', start);
        const std::size_t stop = (pos == std::string::npos) ? s.size() : pos;
        parts.emplace_back(s.data() + start, stop - start);
        if (pos == std::string::npos) break;
        start = pos + 1;
    }
    if (parts.size() % 3 != 0) throw std::runtime_error("malformed Properties string '" + s + "'");

    std::vector<ColumnSpec> specs;
    for (std::size_t k = 0; k < parts.size(); k += 3) {
        ColumnSpec spec;
        spec.name = std::string(parts[k]);
        if (parts[k + 1].size() != 1) throw std::runtime_error("bad column type in Properties '" + s + "'");
        switch (parts[k + 1][0]) {
            case 'R': spec.type = PropType::Real; break;
            case 'I': spec.type = PropType::Int; break;
            case 'L': spec.type = PropType::Bool; break;
            case 'S': spec.type = PropType::Str; break;
            default: throw std::runtime_error("bad column type in Properties '" + s + "'");
        }
        std::int64_t n = 0;
        if (!parse_int(parts[k + 2], n) || n < 1) {
            throw std::runtime_error("bad column count in Properties '" + s + "'");
        }
        spec.ncols = static_cast<int>(n);
        specs.push_back(std::move(spec));
    }
    return specs;
}

std::vector<double> parse_double_list(const std::string& s, const char* what) {
    std::vector<double> vals;
    for (auto tok : tokenize(s)) {
        double v;
        if (!parse_double(tok, v)) {
            throw std::runtime_error(std::string("cannot parse ") + what + " value '" + std::string(tok) + "'");
        }
        vals.push_back(v);
    }
    return vals;
}

enum class Target { Species, Position, AtomicNumber, Array };

struct Column {
    Target target;
    PropType type;
    int ncols;
    PerAtomArray* array;
};

[[noreturn]] void bad_value(std::string_view tok, std::size_t atom) {
    throw std::runtime_error("cannot parse '" + std::string(tok) + "' on atom line " + std::to_string(atom + 1));
}

// ---------------------------------------------------------------- formatting

constexpr double kPow10d[16] = {1e0, 1e1, 1e2,  1e3,  1e4,  1e5,  1e6,  1e7,
                                1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15};
constexpr std::uint64_t kPow10u[16] = {1ULL,
                                       10ULL,
                                       100ULL,
                                       1000ULL,
                                       10000ULL,
                                       100000ULL,
                                       1000000ULL,
                                       10000000ULL,
                                       100000000ULL,
                                       1000000000ULL,
                                       10000000000ULL,
                                       100000000000ULL,
                                       1000000000000ULL,
                                       10000000000000ULL,
                                       100000000000000ULL,
                                       1000000000000000ULL};

// Equivalent to printf("%*.*f") but without the printf overhead for normal-sized values.
void append_fixed(std::string& out, double x, int width, int prec) {
    if (!std::isfinite(x) || std::fabs(x) * kPow10d[prec] >= 9.0e18) {
        const int len = std::snprintf(nullptr, 0, "%*.*f", width, prec, x);
        const std::size_t old = out.size();
        out.resize(old + static_cast<std::size_t>(len) + 1);
        std::snprintf(&out[old], static_cast<std::size_t>(len) + 1, "%*.*f", width, prec, x);
        out.resize(old + static_cast<std::size_t>(len));
        return;
    }
    const bool neg = std::signbit(x);
    const auto v = static_cast<std::uint64_t>(std::llround(std::fabs(x) * kPow10d[prec]));
    std::uint64_t ip = v / kPow10u[prec];
    std::uint64_t fp = v % kPow10u[prec];

    char buf[48];
    char* q = buf + sizeof(buf);
    for (int k = 0; k < prec; ++k) {
        *--q = static_cast<char>('0' + fp % 10);
        fp /= 10;
    }
    if (prec > 0) *--q = '.';
    do {
        *--q = static_cast<char>('0' + ip % 10);
        ip /= 10;
    } while (ip);
    if (neg) *--q = '-';
    const auto len = static_cast<int>(buf + sizeof(buf) - q);
    if (len < width) out.append(static_cast<std::size_t>(width - len), ' ');
    out.append(q, static_cast<std::size_t>(len));
}

// Shortest of %.15g / %.16g / %.17g that round-trips exactly.
void append_general(std::string& out, double x) {
    char buf[40];
    for (int prec = 15; prec <= 17; ++prec) {
        const int len = std::snprintf(buf, sizeof(buf), "%.*g", prec, x);
        if (prec == 17 || std::strtod(buf, nullptr) == x) {
            out.append(buf, static_cast<std::size_t>(len));
            return;
        }
    }
}

bool needs_quotes(const std::string& s) {
    if (s.empty()) return true;
    for (char c : s) {
        if (is_space(c) || c == '=' || c == '"' || c == '\n') return true;
    }
    return false;
}

void append_quoted(std::string& out, const std::string& s) {
    out += '"';
    for (char c : s) {
        if (c == '\n') {
            out += "\\n";
            continue;
        }
        if (c == '"' || c == '\\') out += '\\';
        out += c;
    }
    out += '"';
}

struct InfoFormatter {
    std::string& out;
    void operator()(bool b) const { out += b ? 'T' : 'F'; }
    void operator()(std::int64_t v) const { out += std::to_string(v); }
    void operator()(double v) const { append_general(out, v); }
    void operator()(const std::string& s) const {
        if (needs_quotes(s)) {
            append_quoted(out, s);
        } else {
            out += s;
        }
    }
    void operator()(const std::vector<double>& v) const {
        out += '"';
        for (std::size_t i = 0; i < v.size(); ++i) {
            if (i) out += ' ';
            append_general(out, v[i]);
        }
        out += '"';
    }
};

constexpr std::size_t kWriteBatch = 512;

void format_batch(std::vector<std::string>& bufs, const std::vector<const Atoms*>& frames, std::size_t start) {
    const auto m = static_cast<std::ptrdiff_t>(bufs.size());
    std::vector<std::string> errors(bufs.size());

#ifdef FASTATOMS_HAVE_OPENMP
    #pragma omp parallel for schedule(dynamic, 8) if (m > 1)
#endif
    for (std::ptrdiff_t k = 0; k < m; ++k) {
        const auto uk = static_cast<std::size_t>(k);
        const Atoms& atoms = *frames[start + uk];
        std::string& b = bufs[uk];
        b.clear();
        b.reserve(72 * (atoms.size() + 2));
        try {
            format_extxyz_frame(b, atoms);
        } catch (const std::exception& ex) {
            errors[uk] = "frame " + std::to_string(start + uk) + ": " + ex.what();
        }
    }
    for (const auto& err : errors) {
        if (!err.empty()) throw std::runtime_error(err);
    }
}

}  // namespace

// ======================================================================= reading

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open file '" + path + "'");
    const std::streamsize size = f.tellg();
    if (size < 0) throw std::runtime_error("cannot determine size of '" + path + "'");
    std::string data(static_cast<std::size_t>(size), '\0');
    f.seekg(0);
    if (size > 0 && !f.read(&data[0], size)) throw std::runtime_error("failed to read '" + path + "'");
    return data;
}

std::vector<FrameSpan> scan_extxyz_frames(std::string_view data) {
    std::vector<FrameSpan> spans;
    const char* const begin = data.data();
    const char* const e = begin + data.size();
    const char* p = begin;

    while (p < e) {
        const char* eol = find_eol(p, e);
        const std::string_view line = trim(std::string_view(p, static_cast<std::size_t>(eol - p)));
        if (line.empty()) {
            p = next_line(eol, e);
            continue;
        }
        std::int64_t natoms = 0;
        if (!parse_int(line, natoms) || natoms < 0) {
            throw std::runtime_error("frame " + std::to_string(spans.size()) +
                                     ": expected an atom count but found '" + std::string(line.substr(0, 60)) + "'");
        }
        const char* frame_begin = p;
        p = next_line(eol, e);
        for (std::int64_t k = 0; k < natoms + 1; ++k) {
            if (p >= e) {
                throw std::runtime_error("frame " + std::to_string(spans.size()) + ": file ends before all " +
                                         std::to_string(natoms) + " atom lines were read");
            }
            p = next_line(find_eol(p, e), e);
        }
        spans.push_back({static_cast<std::size_t>(frame_begin - begin), static_cast<std::size_t>(p - begin), natoms});
    }
    return spans;
}

Atoms parse_extxyz_frame(std::string_view text) {
    const char* p = text.data();
    const char* const e = p + text.size();

    while (p < e) {  // tolerate leading blank lines
        const char* eol = find_eol(p, e);
        if (!trim(std::string_view(p, static_cast<std::size_t>(eol - p))).empty()) break;
        p = next_line(eol, e);
    }

    const char* eol = find_eol(p, e);
    std::int64_t natoms = 0;
    if (!parse_int(trim(std::string_view(p, static_cast<std::size_t>(eol - p))), natoms) || natoms < 0) {
        throw std::runtime_error("invalid atom count line");
    }
    p = next_line(eol, e);

    eol = find_eol(p, e);
    const std::vector<KeyValue> kvs = parse_comment(p, eol);
    p = next_line(eol, e);

    Atoms atoms;
    std::vector<ColumnSpec> specs;
    bool have_props = false, have_pbc = false, have_lattice = false;

    for (const auto& kv : kvs) {
        if (iequals(kv.key, "Lattice")) {
            const auto vals = parse_double_list(kv.value, "Lattice");
            if (vals.size() == 9) {
                std::copy(vals.begin(), vals.end(), atoms.cell.begin());
            } else if (vals.size() == 3) {
                atoms.cell = Mat3{vals[0], 0, 0, 0, vals[1], 0, 0, 0, vals[2]};
            } else {
                throw std::runtime_error("Lattice must have 9 (or 3) values");
            }
            have_lattice = true;
        } else if (iequals(kv.key, "Properties")) {
            specs = parse_properties(kv.value);
            have_props = true;
        } else if (iequals(kv.key, "pbc")) {
            const auto toks = tokenize(kv.value);
            bool b[3];
            if (toks.size() == 3) {
                for (int i = 0; i < 3; ++i) {
                    if (!parse_bool(toks[static_cast<std::size_t>(i)], b[i])) throw std::runtime_error("bad pbc value");
                }
            } else if (toks.size() == 1 && parse_bool(toks[0], b[0])) {
                b[1] = b[2] = b[0];
            } else {
                throw std::runtime_error("bad pbc value '" + kv.value + "'");
            }
            atoms.pbc = {b[0], b[1], b[2]};
            have_pbc = true;
        } else {
            atoms.info[kv.key] = infer_info_value(kv);
        }
    }
    if (!have_props) specs = {{"species", PropType::Str, 1}, {"pos", PropType::Real, 3}};
    if (have_lattice && !have_pbc) atoms.pbc = {true, true, true};

    const auto n = static_cast<std::size_t>(natoms);
    atoms.numbers.assign(n, 0);
    atoms.positions.assign(3 * n, 0.0);

    const bool has_species = std::any_of(specs.begin(), specs.end(), [](const ColumnSpec& s) {
        return s.name == "species" && s.type == PropType::Str && s.ncols == 1;
    });

    std::vector<Column> cols;
    cols.reserve(specs.size());
    for (const auto& s : specs) {
        Column c{Target::Array, s.type, s.ncols, nullptr};
        if (s.name == "species" && s.type == PropType::Str && s.ncols == 1) {
            c.target = Target::Species;
        } else if (s.name == "pos" && s.type == PropType::Real && s.ncols == 3) {
            c.target = Target::Position;
        } else if (s.name == "Z" && s.type == PropType::Int && s.ncols == 1 && !has_species) {
            c.target = Target::AtomicNumber;
        } else {
            PerAtomArray& arr = atoms.arrays[s.name];  // std::map nodes are pointer-stable
            arr.type = s.type;
            arr.ncols = s.ncols;
            arr.resize_rows(n);
            c.array = &arr;
        }
        cols.push_back(c);
    }

    std::string_view tok;
    for (std::size_t i = 0; i < n; ++i) {
        if (p >= e) {
            throw std::runtime_error("file ended after " + std::to_string(i) + " of " + std::to_string(n) +
                                     " atom lines");
        }
        eol = find_eol(p, e);
        const char* q = p;
        for (const Column& c : cols) {
            for (int k = 0; k < c.ncols; ++k) {
                if (!next_token(q, eol, tok)) {
                    throw std::runtime_error("atom line " + std::to_string(i + 1) +
                                             " has fewer columns than Properties declares");
                }
                const auto uk = static_cast<std::size_t>(k);
                switch (c.target) {
                    case Target::Species:
                        atoms.numbers[i] = symbol_to_number(tok);
                        break;
                    case Target::Position:
                        if (!parse_double(tok, atoms.positions[3 * i + uk])) bad_value(tok, i);
                        break;
                    case Target::AtomicNumber: {
                        std::int64_t z;
                        if (!parse_int(tok, z) || z < 0 || z > kMaxZ) bad_value(tok, i);
                        atoms.numbers[i] = static_cast<int>(z);
                        break;
                    }
                    case Target::Array: {
                        PerAtomArray& arr = *c.array;
                        const std::size_t idx = i * static_cast<std::size_t>(c.ncols) + uk;
                        switch (c.type) {
                            case PropType::Real:
                                if (!parse_double(tok, arr.real[idx])) bad_value(tok, i);
                                break;
                            case PropType::Int:
                                if (!parse_int(tok, arr.integer[idx])) bad_value(tok, i);
                                break;
                            case PropType::Bool: {
                                bool b;
                                if (!parse_bool(tok, b)) bad_value(tok, i);
                                arr.logical[idx] = b ? 1 : 0;
                                break;
                            }
                            case PropType::Str:
                                arr.str[idx].assign(tok.data(), tok.size());
                                break;
                        }
                        break;
                    }
                }
            }
        }
        p = next_line(eol, e);
    }
    return atoms;
}

std::vector<Atoms> parse_extxyz_frames(std::string_view data, const std::vector<FrameSpan>& spans,
                                       const std::vector<std::size_t>& selection) {
    const auto m = static_cast<std::ptrdiff_t>(selection.size());
    std::vector<Atoms> frames(selection.size());
    std::vector<std::string> errors(selection.size());

#ifdef FASTATOMS_HAVE_OPENMP
    #pragma omp parallel for schedule(dynamic, 8) if (m > 1)
#endif
    for (std::ptrdiff_t k = 0; k < m; ++k) {
        const auto uk = static_cast<std::size_t>(k);
        const FrameSpan& s = spans[selection[uk]];
        try {
            frames[uk] = parse_extxyz_frame(data.substr(s.begin, s.end - s.begin));
        } catch (const std::exception& ex) {
            errors[uk] = "frame " + std::to_string(selection[uk]) + ": " + ex.what();
        }
    }
    for (const auto& err : errors) {
        if (!err.empty()) throw std::runtime_error(err);
    }
    return frames;
}

// ======================================================================= writing

void format_extxyz_frame(std::string& out, const Atoms& a) {
    const std::size_t n = a.size();
    if (a.positions.size() != 3 * n) throw std::runtime_error("positions and numbers have inconsistent sizes");

    out += std::to_string(n);
    out += '\n';

    bool first = true;
    auto sep = [&]() {
        if (!first) out += ' ';
        first = false;
    };

    const bool has_cell = std::any_of(a.cell.begin(), a.cell.end(), [](double v) { return v != 0.0; }) ||
                          a.pbc[0] || a.pbc[1] || a.pbc[2];
    if (has_cell) {
        sep();
        out += "Lattice=\"";
        for (int i = 0; i < 9; ++i) {
            if (i) out += ' ';
            append_general(out, a.cell[static_cast<std::size_t>(i)]);
        }
        out += '"';
    }

    sep();
    out += "Properties=species:S:1:pos:R:3";
    for (const auto& kv : a.arrays) {
        if (kv.second.nrows() != n) {
            throw std::runtime_error("per-atom array '" + kv.first + "' has the wrong length");
        }
        out += ':';
        out += kv.first;
        out += ':';
        out += static_cast<char>(kv.second.type);
        out += ':';
        out += std::to_string(kv.second.ncols);
    }

    for (const auto& kv : a.info) {
        if (iequals(kv.first, "Lattice") || iequals(kv.first, "Properties") || iequals(kv.first, "pbc")) continue;
        sep();
        out += kv.first;
        out += '=';
        std::visit(InfoFormatter{out}, kv.second);
    }

    sep();
    out += "pbc=\"";
    out += a.pbc[0] ? 'T' : 'F';
    out += ' ';
    out += a.pbc[1] ? 'T' : 'F';
    out += ' ';
    out += a.pbc[2] ? 'T' : 'F';
    out += "\"\n";

    for (std::size_t i = 0; i < n; ++i) {
        const char* sym = number_to_symbol(a.numbers[i]);
        const std::size_t len = std::strlen(sym);
        out.append(sym, len);
        if (len < 2) out += ' ';
        for (std::size_t c = 0; c < 3; ++c) {
            out += ' ';
            append_fixed(out, a.positions[3 * i + c], 15, 8);
        }
        for (const auto& kv : a.arrays) {
            const PerAtomArray& arr = kv.second;
            for (int c = 0; c < arr.ncols; ++c) {
                const std::size_t idx = i * static_cast<std::size_t>(arr.ncols) + static_cast<std::size_t>(c);
                out += ' ';
                switch (arr.type) {
                    case PropType::Real: append_fixed(out, arr.real[idx], 15, 8); break;
                    case PropType::Int: out += std::to_string(arr.integer[idx]); break;
                    case PropType::Bool: out += arr.logical[idx] ? 'T' : 'F'; break;
                    case PropType::Str: out += arr.str[idx]; break;
                }
            }
        }
        out += '\n';
    }
}

std::string format_extxyz(const std::vector<const Atoms*>& frames) {
    std::string out;
    std::vector<std::string> bufs;
    for (std::size_t s = 0; s < frames.size(); s += kWriteBatch) {
        bufs.resize(std::min(kWriteBatch, frames.size() - s));
        format_batch(bufs, frames, s);
        for (const auto& b : bufs) out += b;
    }
    return out;
}

void write_extxyz(const std::string& path, const std::vector<const Atoms*>& frames, bool append) {
    std::ofstream f(path, std::ios::binary | (append ? std::ios::app : std::ios::trunc));
    if (!f) throw std::runtime_error("cannot open '" + path + "' for writing");
    std::vector<std::string> bufs;
    for (std::size_t s = 0; s < frames.size(); s += kWriteBatch) {
        bufs.resize(std::min(kWriteBatch, frames.size() - s));
        format_batch(bufs, frames, s);
        for (const auto& b : bufs) f.write(b.data(), static_cast<std::streamsize>(b.size()));
    }
    if (!f) throw std::runtime_error("error while writing '" + path + "'");
}

}  // namespace fastatoms
