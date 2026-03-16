//go:build amd64

package blockvector

import (
	"log/slog"
	"os"

	"golang.org/x/sys/cpu"
)

// Assembly function signatures defined in engine_amd64.s
func DotProductAVX2(a, b []float32) float32
func DotProductInt8AVX2(a, b []int8) int32

func initDispatcher() {
	forceGeneric := os.Getenv("BLOCKVECTOR_FORCE_GENERIC") == "1"

	// Dispatch Float32 engine
	if !forceGeneric && cpu.X86.HasAVX2 && cpu.X86.HasFMA {
		dotProductFn = DotProductAVX2
		slog.Debug("Float32 engine: AVX2+FMA active")
	} else {
		dotProductFn = DotProductGeneric
		slog.Debug("Float32 engine: Generic fallback")
	}

	// Dispatch Int8 engine
	if !forceGeneric && cpu.X86.HasAVX2 {
		dotProductInt8Fn = DotProductInt8AVX2
		slog.Debug("Int8 engine: AVX2 active")
	} else {
		dotProductInt8Fn = DotProductInt8Generic
		slog.Debug("Int8 engine: Generic fallback")
	}
}
