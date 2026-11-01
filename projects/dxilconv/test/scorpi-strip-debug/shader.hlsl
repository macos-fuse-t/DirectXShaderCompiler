cbuffer Parameters : register(b3) { uint value; };
RWStructuredBuffer<uint> output : register(u2);
float4 main(uint id : SV_VertexID) : SV_Position {
    uint result = value + id;
    output[id] = result;
    return float4(result, id, 0.5f, 1.0f);
}
