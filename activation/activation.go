package activation

import (
	"math"

	"github.com/herlon214/go-neural/tensor"
)

type Function func(x tensor.Tensor) tensor.Tensor

func Sigmoid(x tensor.Tensor) tensor.Tensor {
	return tensor.Tensor(1.0 / (1.0 + math.Exp(-x.Float64())))
}

func SigmoidSlope(y tensor.Tensor) tensor.Tensor {
	return y * (1 - y)

}
