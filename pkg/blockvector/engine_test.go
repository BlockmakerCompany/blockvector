package blockvector

import (
	"math/rand"
	"testing"
)

const (
	TestDimensions = 1536
	TestCount      = 10000 // Reducido un poco para que los tests no tarden una eternidad, pero suficiente para saturar cache
)

// --- MOCK DATA GENERATORS ---

func generateFlatMockData(n, dim int) []float32 {
	data := make([]float32, n*dim)
	for i := range data {
		data[i] = rand.Float32()
	}
	return data
}

func generateInt8MockData(n, dim int) []int8 {
	data := make([]int8, n*dim)
	for i := range data {
		data[i] = int8(rand.Intn(256) - 128)
	}
	return data
}

// --- ACCURACY TESTS ---

func TestDotProductConsistency(t *testing.T) {
	// We test with a real dimension size to ensure AVX2 alignment
	v1 := generateInt8MockData(1, TestDimensions)
	v2 := generateInt8MockData(1, TestDimensions)

	// We call the internal functions directly since we are in the same package
	gen := DotProductInt8Generic(v1, v2)
	avx := DotProductInt8AVX2(v1, v2)

	if gen != avx {
		t.Errorf("Int8 Mismatch! Generic: %d, AVX2: %d", gen, avx)
	}
}

// --- BENCHMARKS ---

func BenchmarkLinearScanFloat32_Parallel(b *testing.B) {
	dataset := generateFlatMockData(TestCount, TestDimensions)
	query := make([]float32, TestDimensions)
	copy(query, dataset[42*TestDimensions:])

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Testing the parallel float32 engine
		idx, _ := LinearScanParallel(query, dataset, TestDimensions)
		if idx != 42 {
			b.Fail()
		}
	}
}

func BenchmarkLinearScanInt8_Serial_Generic(b *testing.B) {
	dataset := generateInt8MockData(TestCount, TestDimensions)
	query := dataset[42*TestDimensions : 43*TestDimensions]

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var max int32 = -2147483648
		for j := 0; j < len(dataset); j += TestDimensions {
			// Forced serial call to Generic to see the "slow" baseline
			score := DotProductInt8Generic(query, dataset[j:j+TestDimensions])
			if score > max {
				max = score
			}
		}
	}
}

func BenchmarkLinearScanInt8_Parallel_AVX2(b *testing.B) {
	dataset := generateInt8MockData(TestCount, TestDimensions)
	query := dataset[42*TestDimensions : 43*TestDimensions]

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Testing the top-level parallel engine (will use AVX2 via dispatcher)
		idx, _ := LinearScanInt8Parallel(query, dataset, TestDimensions)
		if idx != 42 {
			b.Fail()
		}
	}
}

func TestTopKAccuracy(t *testing.T) {
	dim := 1536
	k := 5
	n := 100
	dataset := make([]int8, n*dim)
	query := make([]int8, dim)

	// Llenamos con basura
	for i := range dataset {
		dataset[i] = 1
	}
	for i := range query {
		query[i] = 10
	}

	// Insertamos la "aguja" en el pajar (índice 42)
	// Este vector es idéntico al query, debe tener el score máximo
	for i := 0; i < dim; i++ {
		dataset[42*dim+i] = 10
	}

	results := LinearScanInt8TopK(query, dataset, dim, k)

	if len(results) != k {
		t.Errorf("Expected %d results, got %d", k, len(results))
	}

	if results[0].Index != 42 {
		t.Errorf("The best match should be index 42, but got %d", results[0].Index)
	}
}

func TestExtremeValues(t *testing.T) {
	dim := 1536
	// Caso A: Todo ceros (Score debe ser 0)
	a := make([]int8, dim)
	b := make([]int8, dim)
	if DotProductInt8AVX2(a, b) != 0 {
		t.Error("Dot product of zeros should be 0")
	}

	// Caso B: Todo Max (127 * 127 * 1536)
	// Esto verifica que el acumulador de 32 bits no desborde
	for i := range a {
		a[i] = 127
		b[i] = 127
	}
	score := DotProductInt8AVX2(a, b)
	if score <= 0 {
		t.Errorf("Score overflow detected with max values: %d", score)
	}
}

func TestKGreaterThanDataset(t *testing.T) {
	dim := 128 // Dimensión pequeña para variar
	dataset := make([]int8, 3*dim)
	query := make([]int8, dim)
	k := 10 // K es mayor que N

	results := LinearScanInt8TopK(query, dataset, dim, k)

	if len(results) != 3 {
		t.Errorf("Should return 3 results, but got %d", len(results))
	}
}
