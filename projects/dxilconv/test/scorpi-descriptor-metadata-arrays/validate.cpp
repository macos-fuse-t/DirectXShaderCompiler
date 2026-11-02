#include "dxc/dxcapi.h"
#include <cassert>
#include <cstdio>
#include <dlfcn.h>
#include <fstream>
#include <vector>

int main(int argc, char **argv) {
  assert(argc == 3);
  void *library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  assert(library);
  auto create = reinterpret_cast<DxcCreateInstanceProc>(dlsym(library, "DxcCreateInstance"));
  assert(create);
  IDxcLibrary *utils = nullptr;
  IDxcValidator *validator = nullptr;
  assert(SUCCEEDED(create(CLSID_DxcLibrary, IID_PPV_ARGS(&utils))));
  assert(SUCCEEDED(create(CLSID_DxcValidator, IID_PPV_ARGS(&validator))));
  std::ifstream input(argv[2], std::ios::binary | std::ios::ate);
  assert(input && input.tellg() > 0);
  std::vector<char> data(size_t(input.tellg()));
  input.seekg(0); input.read(data.data(), data.size());
  IDxcBlobEncoding *blob = nullptr;
  assert(SUCCEEDED(utils->CreateBlobWithEncodingOnHeapCopy(data.data(), data.size(), 0, &blob)));
  IDxcOperationResult *result = nullptr;
  HRESULT status = validator->Validate(blob, DxcValidatorFlags_Default, &result);
  if (result) {
    result->GetStatus(&status);
    IDxcBlobEncoding *errors = nullptr;
    result->GetErrorBuffer(&errors);
    if (errors) {
      std::printf("%.*s", int(errors->GetBufferSize()),
          static_cast<const char *>(errors->GetBufferPointer()));
      errors->Release();
    }
    result->Release();
  }
  blob->Release(); validator->Release(); utils->Release();
  return FAILED(status) ? 1 : 0;
}
