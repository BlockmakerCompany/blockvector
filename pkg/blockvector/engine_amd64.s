#include "textflag.h"

// func DotProductAVX2(a, b []float32) float32
// Go Slice Header (24 bytes): 0:ptr, 8:len, 16:cap
// Argument offsets: a=0(FP), b=24(FP), ret=48(FP)
// Frame size: $0-52 (48 bytes for 2 slices + 4 bytes for float32 return)
TEXT ·DotProductAVX2(SB), NOSPLIT, $0-52
    // Load slice pointers and length
    MOVQ a_base+0(FP), SI    // SI = pointer to a
    MOVQ a_len+8(FP), CX     // CX = length of a
    MOVQ b_base+24(FP), DI   // DI = pointer to b

    // Clear Y0 for accumulation (8x float32)
    VXORPS Y0, Y0, Y0

loop_f32:
    // Load 32 bytes (8 floats) from each slice
    VMOVUPS (SI), Y1
    VMOVUPS (DI), Y2

    // Y0 = Y0 + (Y1 * Y2) using FMA (Fused Multiply-Add)
    VFMADD231PS Y2, Y1, Y0

    // Advance pointers by 32 bytes (8 floats)
    ADDQ $32, SI
    ADDQ $32, DI
    // Decrease counter by 8 elements
    SUBQ $8, CX
    CMPQ CX, $8
    JGE loop_f32

    // --- Reduction: Sum the 8 partial results in Y0 ---
    // Extract high 128-bit lane from Y0 into X1
    VEXTRACTF128 $1, Y0, X1
    // Add high and low 128-bit lanes: X0 = X0 + X1
    VADDPS X1, X0, X0
    // Horizontal add partial sums
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0

    // Return the final single-precision result
    MOVSS X0, ret+48(FP)
    RET

// func DotProductInt8AVX2(a, b []int8) int32
// Argument offsets: a=0(FP), b=24(FP), ret=48(FP)
// Frame size: $0-52 (48 bytes for 2 slices + 4 bytes for int32 return)
TEXT ·DotProductInt8AVX2(SB), NOSPLIT, $0-52
    // Load slice pointers and length
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX
    MOVQ b_base+24(FP), DI

    // Clear Y0 for accumulation (8x int32)
    VPXOR Y0, Y0, Y0

loop_int8:
    // Load 16 bytes (16 elements) from each slice
    VMOVDQU (SI), X2
    VMOVDQU (DI), X3

    // Step 1: Sign-extend 16x int8 to 16x int16 (words)
    VPMOVSXBW X2, Y4
    VPMOVSXBW X3, Y5

    // Step 2: Multiply words and horizontally add pairs into dwords (int32)
    // Result: 8x int32 in Y6
    VPMADDWD Y4, Y5, Y6

    // Step 3: Accumulate the 8 dwords into Y0
    VPADDD Y6, Y0, Y0

    // Advance pointers by 16 bytes
    ADDQ $16, SI
    ADDQ $16, DI
    // Decrease counter by 16 elements
    SUBQ $16, CX
    CMPQ CX, $16
    JGE loop_int8

    // --- Reduction: Sum the 8 partial int32 results in Y0 ---
    VEXTRACTI128 $1, Y0, X1  // Get high 128 bits
    VPADDD X1, X0, X0        // Add to low 128 bits
    VPSHUFD $0x4E, X0, X1    // Shuffle partial sums
    VPADDD X1, X0, X0
    VPSHUFD $0xB1, X0, X1
    VPADDD X1, X0, X0

    // Return the final int32 result
    MOVL X0, ret+48(FP)
    RET
