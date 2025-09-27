package activation

import (
	"math"

	"github.com/herlon214/go-neural/tensor"
)

type Sigmoid struct{}

func NewSigmoid() Sigmoid {
	return Sigmoid{}
}

func (s Sigmoid) Activate(x tensor.Tensor) tensor.Tensor {
	return tensor.Tensor(1.0 / (1.0 + math.Exp(-x.Float64())))
}

func (s Sigmoid) Derive(y tensor.Tensor) tensor.Tensor {
	return y * (1 - y)
}
