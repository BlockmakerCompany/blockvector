//go:build arm64

#include "textflag.h"

// func DotProductNEON(a, b []float32) float32
// Signature: (a_ptr, a_len, a_cap, b_ptr, b_len, b_cap)
// Registers:
// R0: a_ptr | R1: a_len
// R3: b_ptr | F0: return value (float32)
TEXT ·DotProductNEON(SB), NOSPLIT, $0
    MOVD    a_base+0(FP), R0
    MOVD    b_base+24(FP), R3
    MOVD    a_len+8(FP), R1

    // Clear V0 accumulator (128-bit / 4x float32)
    VEOR    V0.B16, V0.B16, V0.B16

loop_f32:
    CMP     $4, R1
    BLT     done_f32

    // Load 4 floats (128 bits) from each slice with post-increment
    VLD1.P  16(R0), [V1.S4]
    VLD1.P  16(R3), [V2.S4]

    // Fused Multiply-Add: V0 = V0 + (V1 * V2)
    VFMLA.S4 V0, V1, V2

    SUB     $4, R1
    B       loop_f32

done_f32:
    // Horizontal reduction: sum the 4 float lanes in V0
    FADDP.S4 V0, V0, V0
    FADDP.S2 V0, V0, V0
    FMOV    S0, F0    // Move final result to return register
    RET

// func DotProductInt8NEON(a, b []int8) int32
// Signature: (a_ptr, a_len, a_cap, b_ptr, b_len, b_cap)
// Registers:
// R0: a_ptr | R1: a_len
// R3: b_ptr | R0: return value (int32)
TEXT ·DotProductInt8NEON(SB), NOSPLIT, $0
    MOVD    a_base+0(FP), R0
    MOVD    b_base+24(FP), R3
    MOVD    a_len+8(FP), R1

    // Clear V0 accumulator (4x int32 lanes)
    VEOR    V0.B16, V0.B16, V0.B16

loop_int8:
    CMP     $16, R1
    BLT     done_int8

    // Load 16 bytes (int8) into V1 and V2
    VLD1.P  16(R0), [V1.B16]
    VLD1.P  16(R3), [V2.B16]

    // Long Multiplication: int8 * int8 -> int16
    // SMULL: Multiply lower 8 bytes
    // SMULL2: Multiply upper 8 bytes
    SMULL.B8 V3, V1, V2
    SMULL2.B16 V4, V1, V2

    // Accumulate: int32 = int32 + int16
    SADDW.H4 V0, V0, V3
    SADDW2.H8 V0, V0, V4

    SUB     $16, R1
    B       loop_int8

done_int8:
    // Horizontal reduction of the 4x int32 lanes in V0
    ADDV.S4 S0, V0
    FMOV    S0, R0    // Move result to R0 for return
    RET
