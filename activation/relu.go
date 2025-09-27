package activation

import "github.com/herlon214/go-neural/tensor"

type ReLU struct{}

func NewReLU() ReLU {
	return ReLU{}
}

func (r ReLU) Activate(x tensor.Tensor) tensor.Tensor {
	if x > 0 {
		return x
	}

	return 0
}

func (r ReLU) Derive(y tensor.Tensor) tensor.Tensor {
	if y > 0 {
		return 1
	}

	return 0
}
