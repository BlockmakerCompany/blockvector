package main

/*
#include <stdlib.h>

// Definimos la estructura Result para que sea compatible con C
typedef struct {
    int index;
    int score;
} Result;
*/
import "C"
import (
	"github.com/BlockmakerCompany/blockvector/pkg/blockvector"

	"unsafe"
)

//export LinearScanInt8TopK
func LinearScanInt8TopK(queryPtr *C.schar, datasetPtr *C.schar, queryLen C.int, datasetLen C.int, dim C.int, k C.int) *C.Result {
	// 1. Convertimos los punteros de C a slices de Go (Zero-copy)
	query := unsafe.Slice((*int8)(queryPtr), int(queryLen))
	dataset := unsafe.Slice((*int8)(datasetPtr), int(datasetLen))

	// 2. Ejecutamos el motor de búsqueda de BlockVector
	goResults := blockvector.LinearScanInt8TopK(query, dataset, int(dim), int(k))

	// 3. Reservamos memoria en el heap de C para devolver los resultados
	// El llamador (C/Rust/Python) será responsable de hacer free()
	resultSize := unsafe.Sizeof(C.Result{})
	cArray := C.malloc(C.size_t(len(goResults)) * C.size_t(resultSize))

	// 4. Copiamos los resultados al array de C
	cSlice := unsafe.Slice((*C.Result)(cArray), len(goResults))
	for i, res := range goResults {
		cSlice[i].index = C.int(res.Index)
		cSlice[i].score = C.int(res.Score)
	}

	return (*C.Result)(cArray)
}

// Es obligatorio tener un main vacío para compilar como c-shared
func main() {}
