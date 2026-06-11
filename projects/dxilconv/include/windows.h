#pragma once

#include "dxc/Support/WinIncludes.h"

#ifndef interface
#define interface struct
#endif
#ifndef WINAPI
#define WINAPI
#endif
#ifndef __override
#define __override
#endif

#ifndef _Inout_
#define _Inout_
#endif
#ifndef _Inout_opt_
#define _Inout_opt_
#endif
#ifndef _Inout_updates_bytes_
#define _Inout_updates_bytes_(x)
#endif
#ifndef _In_reads_bytes_
#define _In_reads_bytes_(x)
#endif
#ifndef _In_reads_bytes_opt_
#define _In_reads_bytes_opt_(x)
#endif
#ifndef _Out_writes_bytes_opt_
#define _Out_writes_bytes_opt_(x)
#endif
#ifndef _Out_writes_bytes_
#define _Out_writes_bytes_(x)
#endif
#ifndef _Out_writes_
#define _Out_writes_(x)
#endif
#ifndef _Out_writes_opt_
#define _Out_writes_opt_(x)
#endif
#ifndef _In_reads_
#define _In_reads_(x)
#endif
#ifndef _In_reads_opt_
#define _In_reads_opt_(x)
#endif
#ifndef _In_range_
#define _In_range_(low, high)
#endif
#ifndef __in_range
#define __in_range(low, high)
#endif
#ifndef __in_ecount
#define __in_ecount(x)
#endif
#ifndef __out_ecount
#define __out_ecount(x)
#endif
#ifndef __field_ecount_part
#define __field_ecount_part(size, count)
#endif
#ifndef _Inexpressible_
#define _Inexpressible_(x)
#endif
#ifndef _Field_size_bytes_full_
#define _Field_size_bytes_full_(x)
#endif
#ifndef _Field_size_bytes_full_opt_
#define _Field_size_bytes_full_opt_(x)
#endif
#ifndef _Field_size_full_
#define _Field_size_full_(x)
#endif
#ifndef _Field_size_
#define _Field_size_(x)
#endif
#ifndef _Always_
#define _Always_(x) x
#endif
#ifndef _Outptr_opt_result_maybenull_
#define _Outptr_opt_result_maybenull_
#endif
#ifndef _Outptr_opt_result_bytebuffer_
#define _Outptr_opt_result_bytebuffer_(x)
#endif
#ifndef _Outptr_result_bytebuffer_maybenull_
#define _Outptr_result_bytebuffer_maybenull_(x)
#endif
#ifndef _Outptr_result_maybenull_z_
#define _Outptr_result_maybenull_z_
#endif
#ifndef DLL_PROCESS_ATTACH
#define DLL_PROCESS_ATTACH 1
#endif
#ifndef DLL_PROCESS_DETACH
#define DLL_PROCESS_DETACH 0
#endif
#ifndef __analysis_assume
#define __analysis_assume(x)
#endif
#ifndef CONST
#define CONST const
#endif
#ifndef _UI8_MAX
#define _UI8_MAX 0xff
#endif
#ifndef D3D11_COMMONSHADER_TEMP_REGISTER_COUNT
#define D3D11_COMMONSHADER_TEMP_REGISTER_COUNT 4096
#endif
#ifndef D3D10_REQ_CONSTANT_BUFFER_ELEMENT_COUNT
#define D3D10_REQ_CONSTANT_BUFFER_ELEMENT_COUNT 4096
#endif
#ifndef D3D11_SHADER_MAX_INTERFACES
#define D3D11_SHADER_MAX_INTERFACES 253
#endif
#ifndef D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT
#define D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT 14
#endif
#ifndef D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT
#define D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT 16
#endif
#ifndef D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT
#define D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT 128
#endif

typedef float FLOAT;
typedef int64_t INT64;
typedef uint16_t UINT16;
typedef GUID UUID;
typedef void *HWND;
typedef struct _SECURITY_ATTRIBUTES {
  DWORD nLength;
  LPVOID lpSecurityDescriptor;
  BOOL bInheritHandle;
} SECURITY_ATTRIBUTES;
typedef struct _LUID {
  DWORD LowPart;
  LONG HighPart;
} LUID;
typedef struct tagRECT {
  long left;
  long top;
  long right;
  long bottom;
} RECT;

inline HANDLE GetProcessHeap() { return nullptr; }
inline void *HeapAlloc(HANDLE, DWORD, SIZE_T bytes) { return malloc(bytes); }
inline BOOL HeapFree(HANDLE, DWORD, void *ptr) {
  free(ptr);
  return TRUE;
}
inline BOOL QueryPerformanceCounter(LARGE_INTEGER *value) {
  if (value != nullptr)
    value->QuadPart = 0;
  return TRUE;
}
inline HRESULT StringCchCopyA(char *dst, size_t dst_count, const char *src) {
  if (dst == nullptr || dst_count == 0)
    return E_INVALIDARG;
  if (src == nullptr) {
    dst[0] = '\0';
    return S_OK;
  }
  strncpy(dst, src, dst_count);
  dst[dst_count - 1] = '\0';
  return S_OK;
}
