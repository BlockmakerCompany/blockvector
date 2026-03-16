package blockvector

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"syscall"
	"unsafe"
)

// MapDataset abre un archivo binario y lo mapea en el espacio de direcciones del proceso.
// Proporciona acceso zero-copy a los datos de los vectores.
func MapDataset(fileName string) ([]int8, *os.File, error) {
	file, err := os.OpenFile(fileName, os.O_RDONLY, 0644)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to open file: %w", err)
	}

	info, err := file.Stat()
	if err != nil {
		file.Close()
		return nil, nil, fmt.Errorf("failed to stat file: %w", err)
	}

	size := info.Size()
	if size == 0 {
		file.Close()
		return nil, nil, fmt.Errorf("file is empty")
	}

	// Mmap: Mapea el archivo directamente en la memoria virtual.
	data, err := syscall.Mmap(int(file.Fd()), 0, int(size), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		file.Close()
		return nil, nil, fmt.Errorf("mmap failed: %w", err)
	}

	// Convertimos el slice de bytes a int8 sin copiar datos.
	int8Data := *(*[]int8)(unsafe.Pointer(&data))

	return int8Data, file, nil
}

// SaveDataset guarda el dataset cuantizado en un archivo binario.
// Crea automáticamente los directorios necesarios si no existen.
func SaveDataset(fileName string, data []int8) error {
	// 1. Garantizar que el directorio exista (ej: "data/")
	dir := filepath.Dir(fileName)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create directory tree %s: %w", dir, err)
	}

	// 2. Conversión Zero-copy de []int8 a []byte para la escritura
	// Esto evita duplicar el dataset en RAM antes de escribir a disco
	byteData := *(*[]byte)(unsafe.Pointer(&data))

	return os.WriteFile(fileName, byteData, 0644)
}

// Quantize convierte un vector float32 a int8 usando Max-Absolute scaling.
func Quantize(v []float32) []int8 {
	var maxAbs float32
	for _, val := range v {
		absVal := float32(math.Abs(float64(val)))
		if absVal > maxAbs {
			maxAbs = absVal
		}
	}

	quantized := make([]int8, len(v))
	if maxAbs == 0 {
		return quantized
	}

	scale := 127.0 / maxAbs
	for i, val := range v {
		quantized[i] = int8(val * scale)
	}

	return quantized
}
