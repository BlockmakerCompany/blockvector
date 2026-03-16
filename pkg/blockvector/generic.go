package blockvector

// DotProductGeneric provides a platform-independent float32 implementation.
// This is used as a fallback when no SIMD instructions (AVX2/NEON) are available.
func DotProductGeneric(a, b []float32) float32 {
	var total float32
	// Standard loop. The Go compiler might try to auto-vectorize this,
	// but it will never match our hand-written Assembly.
	for i := range a {
		total += a[i] * b[i]
	}
	return total
}

// DotProductInt8Generic provides a platform-independent int8 implementation.
// Useful for testing or running on architectures without specific optimizations.
func DotProductInt8Generic(a, b []int8) int32 {
	var total int32
	for i := range a {
		// We cast to int32 before multiplying to prevent overflow
		// during the intermediate sum.
		total += int32(a[i]) * int32(b[i])
	}
	return total
}
