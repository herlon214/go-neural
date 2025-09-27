package network

import (
	"fmt"
	"math/rand"

	"github.com/herlon214/go-neural/activation"
	"github.com/herlon214/go-neural/tensor"
)

type Neuron struct {
	weights      tensor.Tensors
	bias         tensor.Tensor
	activation   activation.Activation
	learningRate tensor.Tensor
}

func NewNeuron(size int, learningRate tensor.Tensor, activation activation.Activation) *Neuron {
	weights := make(tensor.Tensors, 0, size)

	for range size {
		weights = append(weights, tensor.Tensor(rand.Float64()-0.5)) // -0.5 to 0.5
	}

	neuron := Neuron{
		weights:      weights,
		bias:         tensor.Tensor(rand.Float64() - 0.5), // -0.5 to 0.5
		activation:   activation,
		learningRate: learningRate,
	}

	return &neuron
}

func (n *Neuron) Forward(inputs tensor.Tensors) tensor.Tensor {
	output := tensor.Tensor(0.0)

	for i := range len(inputs) {
		output += inputs[i] * n.weights[i]
	}

	// Add bias
	output += n.bias

	return n.activation.Activate(output)
}

func (n *Neuron) BackPropagation(errVal tensor.Tensor, inputs tensor.Tensors, output tensor.Tensor) tensor.Tensors {
	slope := n.activation.Derive(output)

	// Calculate gradients
	gradients := make(tensor.Tensors, 0, len(n.weights))
	for i := range len(inputs) {
		gradients = append(gradients, errVal*inputs[i]*slope)
	}
	biasGradient := errVal * slope

	// Update weights
	for i, gradient := range gradients {
		n.weights[i] = n.weights[i] + (gradient * n.learningRate)
	}

	// Update bias
	n.bias = n.bias + (biasGradient * n.learningRate)

	// Calculate errors to send backwards
	errorsToSendBack := make(tensor.Tensors, 0, len(n.weights))
	for _, weight := range n.weights {
		errorsToSendBack = append(errorsToSendBack, errVal*weight*slope)
	}

	return errorsToSendBack
}

func (n *Neuron) String() string {
	weights := "["
	for i, weight := range n.weights {
		if i > 0 {
			weights += ", "
		}
		weights += fmt.Sprintf("%.4f", weight)
	}
	weights += "]"

	return fmt.Sprintf(
		"Neuron{weights=%s, bias=%.4f, activation=Sigmoid, lr=%.4f}",
		weights, n.bias, n.learningRate,
	)
}
