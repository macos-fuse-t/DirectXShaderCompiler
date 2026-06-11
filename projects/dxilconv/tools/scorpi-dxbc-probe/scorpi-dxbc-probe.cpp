#include "ScorpiDxbcToDxil.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<uint8_t> ReadFile(const std::string &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("failed to open " + path);
  }

  stream.seekg(0, std::ios::end);
  const std::streamoff size = stream.tellg();
  stream.seekg(0, std::ios::beg);
  if (size < 0) {
    throw std::runtime_error("failed to size " + path);
  }

  std::vector<uint8_t> data(static_cast<size_t>(size));
  if (!data.empty()) {
    stream.read(reinterpret_cast<char *>(data.data()),
                static_cast<std::streamsize>(data.size()));
    if (!stream) {
      throw std::runtime_error("failed to read " + path);
    }
  }
  return data;
}

void Usage(const char *name) {
  std::fprintf(stderr,
               "usage: %s <shader_code.bin> <shader_signatures.bin> "
               "<output.dxil>\n",
               name);
  std::fprintf(stderr, "       use '-' for <shader_signatures.bin> when the "
                       "shader has no DDI signatures\n");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 4) {
    Usage(argv[0]);
    return 2;
  }

  try {
    const std::vector<uint8_t> code = ReadFile(argv[1]);
    const bool has_signature_file = std::string(argv[2]) != "-";
    const std::vector<uint8_t> signatures =
        has_signature_file ? ReadFile(argv[2]) : std::vector<uint8_t>();

    char error[1024] = {};
    const int status = ScorpiCreateDxilBinaryFile(
        code.data(), code.size(),
        has_signature_file ? signatures.data() : nullptr,
        has_signature_file ? signatures.size() : 0, argv[3], error,
        sizeof(error));
    if (status != SCORPI_DXBC_TO_DXIL_OK) {
      std::fprintf(stderr, "ScorpiCreateDxilBinaryFile failed status=%d: %s\n",
                   status, error);
      return 1;
    }

    std::printf("wrote %s\n", argv[3]);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "error: %s\n", error.what());
    return 1;
  }
}
