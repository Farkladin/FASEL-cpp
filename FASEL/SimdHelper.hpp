#ifndef SIMD_HELPER_HPP
#define SIMD_HELPER_HPP

#include <simd/simd.h>

inline float length(simd_float2 v) { return simd_length(v); }
inline float distance(simd_float2 a, simd_float2 b) { return simd_distance(a, b); }
inline simd_float2 normalize(simd_float2 v) { return simd_normalize(v); }
inline float dot(simd_float2 a, simd_float2 b) { return simd_dot(a, b); }

inline float length(simd_float4 v) { return simd_length(v); }
inline float distance(simd_float4 a, simd_float4 b) { return simd_distance(a, b); }
inline simd_float4 normalize(simd_float4 v) { return simd_normalize(v); }
inline float dot(simd_float4 a, simd_float4 b) { return simd_dot(a, b); }

inline float length3(simd_float4 v) {
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}
inline float distance3(simd_float4 a, simd_float4 b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}
inline simd_float4 normalize3(simd_float4 v) {
    float len = length3(v);
    if (len > 1e-9f) {
        return simd_make_float4(v.x / len, v.y / len, v.z / len, 0.0f);
    }
    return simd_make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}

#endif // SIMD_HELPER_HPP
