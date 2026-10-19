#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "atoms.hpp"

namespace fastatoms {

// Byte range [begin, end) of one frame inside a file buffer.
struct FrameSpan {
    std::size_t begin;
    std::size_t end;
    std::int64_t natoms;
};

std::string read_file(const std::string& path);

// Cheap pass that only locates frame boundaries (atom count line + natoms + 1 lines).
std::vector<FrameSpan> scan_extxyz_frames(std::string_view data);

// Full parse of one frame's text.
Atoms parse_extxyz_frame(std::string_view frame_text);

// Parse the selected frames (indices into spans). Runs in parallel if built with OpenMP.
std::vector<Atoms> parse_extxyz_frames(std::string_view data, const std::vector<FrameSpan>& spans,
                                       const std::vector<std::size_t>& selection);

void format_extxyz_frame(std::string& out, const Atoms& atoms);
std::string format_extxyz(const std::vector<const Atoms*>& frames);
void write_extxyz(const std::string& path, const std::vector<const Atoms*>& frames, bool append);

}  // namespace fastatoms
