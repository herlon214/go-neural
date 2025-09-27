package network

import (
	"fmt"

	"github.com/herlon214/go-neural/activation"
	"github.com/herlon214/go-neural/tensor"
)

type Layer struct {
	neurons []*Neuron
}

func NewLayer(inputSize int, totalNeurons int, learningRate tensor.Tensor, activation activation.Activation) *Layer {
	neurons := make([]*Neuron, 0, totalNeurons)
	for range totalNeurons {
		neurons = append(neurons, NewNeuron(inputSize, learningRate, activation))
	}

	return &Layer{
		neurons: neurons,
	}
}

func (l *Layer) FeedForward(inputs tensor.Tensors) tensor.Tensors {
	results := make(tensor.Tensors, 0, len(l.neurons))

	for _, neuron := range l.neurons {
		results = append(results, neuron.Forward(inputs))
	}

	return results
}

func (l *Layer) Size() int {
	return len(l.neurons)
}

func (l *Layer) BackPropagation(errValues tensor.Tensors, inputs tensor.Tensors, outputs tensor.Tensors) tensor.Tensors {
	errorsPrevLayer := make(tensor.Tensors, len(inputs))
	for i := range len(inputs) {
		errorsPrevLayer[i] = 0
	}

	for i, neuron := range l.neurons {
		backErrors := neuron.BackPropagation(errValues[i], inputs, outputs[i])
		errorsPrevLayer.Sum(backErrors)
	}

	return errorsPrevLayer
}

func (l *Layer) String() string {
	neuronsStr := "["
	for i, neuron := range l.neurons {
		if i > 0 {
			neuronsStr += ", "
		}
		neuronsStr += neuron.String()
	}
	neuronsStr += "]"

	return fmt.Sprintf("HiddenLayer{neurons=%d, %s}",
		len(l.neurons), neuronsStr)
}
