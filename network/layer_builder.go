package network

import (
	"github.com/herlon214/go-neural/activation"
	"github.com/herlon214/go-neural/tensor"
)

type LayerBuilder struct {
	inputSize    int
	totalNeurons int
	activation   activation.Activation
}

func NewLayerBuilder() *LayerBuilder {
	return &LayerBuilder{}
}

func (lb *LayerBuilder) SetNeurons(val int) *LayerBuilder {
	lb.totalNeurons = val

	return lb
}

func (lb *LayerBuilder) SetActivation(fn activation.Activation) *LayerBuilder {
	lb.activation = fn

	return lb
}

func (lb *LayerBuilder) Build(inputSize int, learningRate tensor.Tensor) *Layer {
	return NewLayer(inputSize, lb.totalNeurons, learningRate, lb.activation)
}
