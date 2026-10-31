#ifndef VALUE_TYPE
#define VALUE_TYPE float4
#endif
#ifndef INTERPOLATION
#define INTERPOLATION nointerpolation
#endif
struct Varying { float4 position : SV_Position; INTERPOLATION VALUE_TYPE value : TEXCOORD0; };
Varying VSMain(float2 position : POSITION, VALUE_TYPE value : TEXCOORD0) {
    Varying result; result.position = float4(position, 0, 1); result.value = value; return result;
}
#ifdef NO_POSITION
float4 PSMain(INTERPOLATION VALUE_TYPE value : TEXCOORD0) : SV_Target { return float4(value); }
#else
float4 PSMain(Varying value) : SV_Target { return float4(value.value); }
#endif
