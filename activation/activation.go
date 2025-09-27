package activation

import "github.com/herlon214/go-neural/tensor"

type Activation interface {
	Activate(tensor tensor.Tensor) tensor.Tensor
	Derive(tensor tensor.Tensor) tensor.Tensor
}
