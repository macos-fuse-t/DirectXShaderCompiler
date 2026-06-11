#pragma once

#ifndef _WIN32
#define EventRegisterMicrosoft_Windows_DxcRuntime_API()
#define EventUnregisterMicrosoft_Windows_DxcRuntime_API()
#define DxcRuntimeEtw_DxcRuntimeInitialization_Start()
#define DxcRuntimeEtw_DxcRuntimeInitialization_Stop(hr)
#define DxcRuntimeEtw_DxcRuntimeShutdown_Start()
#define DxcRuntimeEtw_DxcRuntimeShutdown_Stop(hr)
#define DxcRuntimeEtw_DxcTranslate_Start()
#define DxcRuntimeEtw_DxcTranslate_Stop(hr)
#define DxcRuntimeEtw_DxcTranslate_TranslateStats(...)
#endif
