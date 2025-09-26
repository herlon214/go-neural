package tensor

type Tensor float64
type Tensors []Tensor

func (t Tensor) Float64() float64 {
	return float64(t)
}

func (t Tensors) Multiply() Tensor {
	result := t[0]

	for _, val := range t[1:] {
		result *= val
	}

	return result
}

func (t Tensors) Sum(others Tensors) {
	for i, val := range others {
		t[i] += val
	}
}
