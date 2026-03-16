//go:build arm64

package blockvector

import (
	"log/slog"
	"os"

	"golang.org/x/sys/cpu"
)

// Assembly function signatures defined in engine_arm64.s
func DotProductNEON(a, b []float32) float32
func DotProductInt8NEON(a, b []int8) int32

func initDispatcher() {
	forceGeneric := os.Getenv("BLOCKVECTOR_FORCE_GENERIC") == "1"

	// ASIMD is the official ARM name for NEON instructions
	if !forceGeneric && cpu.ARM64.HasASIMD {
		dotProductFn = DotProductNEON
		dotProductInt8Fn = DotProductInt8NEON
		slog.Debug("Engine initialized: ARM NEON (ASIMD) active")
	} else {
		dotProductFn = DotProductGeneric
		dotProductInt8Fn = DotProductInt8Generic
		slog.Debug("Engine initialized: Generic fallback")
	}
}
