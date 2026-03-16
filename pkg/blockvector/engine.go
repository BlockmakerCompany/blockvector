package blockvector

import (
	"container/heap"
	"runtime"
	"sync"
)

// Internal function pointers dispatched at runtime based on CPU architecture.
var dotProductFn func(a, b []float32) float32
var dotProductInt8Fn func(a, b []int8) int32

func init() {
	// initDispatcher is defined in architecture-specific files (dispatch_*.go)
	initDispatcher()
}

// --- TOP-K STRUCTURES ---

// Result represents a matched vector in the dataset.
type Result struct {
	Index int
	Score int32
}

// ResultHeap implements heap.Interface to create a Min-Heap of Results.
type ResultHeap []Result

func (h ResultHeap) Len() int           { return len(h) }
func (h ResultHeap) Less(i, j int) bool { return h[i].Score < h[j].Score } // Min-Heap
func (h ResultHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *ResultHeap) Push(x any)        { *h = append(*h, x.(Result)) }
func (h *ResultHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

// --- SEARCH ENGINE FUNCTIONS ---

// LinearScanInt8TopK performs a parallel search returning the best K results.
func LinearScanInt8TopK(query []int8, dataset []int8, dim int, k int) []Result {
	numWorkers := runtime.NumCPU()
	chunkSize := (len(dataset) / dim / numWorkers) * dim

	workerResults := make([]ResultHeap, numWorkers)
	var wg sync.WaitGroup

	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func(wID int) {
			defer wg.Done()
			start, end := wID*chunkSize, (wID+1)*chunkSize
			if wID == numWorkers-1 {
				end = len(dataset)
			}

			h := &ResultHeap{}
			heap.Init(h)

			for j := start; j < end; j += dim {
				score := dotProductInt8Fn(query, dataset[j:j+dim])

				if h.Len() < k {
					heap.Push(h, Result{Index: j / dim, Score: score})
				} else if score > (*h)[0].Score {
					heap.Pop(h)
					heap.Push(h, Result{Index: j / dim, Score: score})
				}
			}
			workerResults[wID] = *h
		}(i)
	}
	wg.Wait()

	finalHeap := &ResultHeap{}
	heap.Init(finalHeap)
	for _, localHeap := range workerResults {
		for _, res := range localHeap {
			if finalHeap.Len() < k {
				heap.Push(finalHeap, res)
			} else if res.Score > (*finalHeap)[0].Score {
				heap.Pop(finalHeap)
				heap.Push(finalHeap, res)
			}
		}
	}

	finalResults := make([]Result, finalHeap.Len())
	for i := len(finalResults) - 1; i >= 0; i-- {
		finalResults[i] = heap.Pop(finalHeap).(Result)
	}
	return finalResults
}

// LinearScanParallel performs a parallel search for float32 vectors (Top-1).
func LinearScanParallel(query []float32, dataset []float32, dim int) (int, float32) {
	numWorkers := runtime.NumCPU()
	chunkSize := (len(dataset) / dim / numWorkers) * dim
	results := make([]struct {
		idx   int
		score float32
	}, numWorkers)

	var wg sync.WaitGroup
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func(wID int) {
			defer wg.Done()
			start, end := wID*chunkSize, (wID+1)*chunkSize
			if wID == numWorkers-1 {
				end = len(dataset)
			}
			localIdx, maxScore := -1, float32(-1.0)
			for j := start; j < end; j += dim {
				score := dotProductFn(query, dataset[j:j+dim])
				if score > maxScore {
					maxScore, localIdx = score, j/dim
				}
			}
			results[wID] = struct {
				idx   int
				score float32
			}{localIdx, maxScore}
		}(i)
	}
	wg.Wait()

	finalIdx, finalMax := -1, float32(-1.0)
	for _, r := range results {
		if r.score > finalMax {
			finalMax, finalIdx = r.score, r.idx
		}
	}
	return finalIdx, finalMax
}

// LinearScanInt8Parallel performs a parallel search for int8 quantized vectors (Top-1).
func LinearScanInt8Parallel(query []int8, dataset []int8, dim int) (int, int32) {
	numWorkers := runtime.NumCPU()
	chunkSize := (len(dataset) / dim / numWorkers) * dim
	results := make([]struct {
		idx   int
		score int32
	}, numWorkers)

	var wg sync.WaitGroup
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func(wID int) {
			defer wg.Done()
			start, end := wID*chunkSize, (wID+1)*chunkSize
			if wID == numWorkers-1 {
				end = len(dataset)
			}
			localIdx, maxScore := -1, int32(-2147483648)
			for j := start; j < end; j += dim {
				score := dotProductInt8Fn(query, dataset[j:j+dim])
				if score > maxScore {
					maxScore, localIdx = score, j/dim
				}
			}
			results[wID] = struct {
				idx   int
				score int32
			}{localIdx, maxScore}
		}(i)
	}
	wg.Wait()

	finalIdx, finalMax := -1, int32(-2147483648)
	for _, r := range results {
		if r.score > finalMax {
			finalMax, finalIdx = r.score, r.idx
		}
	}
	return finalIdx, finalMax
}
