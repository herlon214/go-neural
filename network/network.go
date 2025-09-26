package network

import (
	"fmt"
	"slices"

	"github.com/herlon214/go-neural/tensor"
)

type Network struct {
	hiddenLayers []*Layer
	outputLayer  *Layer
}

func New(inputSize int, hiddenLayersSize []int, outputLayerSize int, learningRate tensor.Tensor) *Network {
	// Hidden layer
	hiddenLayers := make([]*Layer, 0, len(hiddenLayersSize))
	for i, val := range hiddenLayersSize {
		// Next layer uses the previous layer size
		if i > 0 {
			inputSize = hiddenLayers[i-1].Size()
		}

		hiddenLayers = append(hiddenLayers, NewLayer(inputSize, val, learningRate))
	}

	// Output layer
	outputLayerInputSize := hiddenLayers[len(hiddenLayers)-1].Size()

	return &Network{
		hiddenLayers: hiddenLayers,
		outputLayer:  NewLayer(outputLayerInputSize, outputLayerSize, learningRate),
	}
}

func (n *Network) FeedForward(input tensor.Tensors) tensor.Tensors {
	// Feed hidden layers
	outputs := make([]tensor.Tensors, 0, len(n.hiddenLayers))
	inputs := make([]tensor.Tensors, 0, len(n.hiddenLayers)+1)
	inputs = append(inputs, input)

	for i, layer := range n.hiddenLayers {
		output := layer.FeedForward(inputs[i])
		outputs = append(outputs, output)
		inputs = append(inputs, output)
	}

	// Feed outputs layer
	finalOutput := n.outputLayer.FeedForward(inputs[len(inputs)-1])

	// Calculate loss
	loss := make(tensor.Tensors, 0, len(finalOutput))
	for _, val := range finalOutput {
		loss = append(loss, 1-val)
	}

	// Back propagation on the output layer
	outputErr := n.outputLayer.BackPropagation(loss, inputs[len(inputs)-1], finalOutput)

	for i, layer := range slices.Backward(n.hiddenLayers) {
		outputErr = layer.BackPropagation(outputErr, inputs[i], outputs[i])
	}

	return loss
}

func (n *Network) String() string {
	architecture := "["
	for i, layer := range n.hiddenLayers {
		if i > 0 {
			architecture += ", "
		}
		architecture += fmt.Sprintf("Hidden%d(%d neurons)", i, len(layer.neurons))
	}
	if len(n.hiddenLayers) > 0 {
		architecture += ", "
	}
	architecture += fmt.Sprintf("Output(%d neurons)", len(n.outputLayer.neurons))
	architecture += "]"

	totalLayers := len(n.hiddenLayers) + 1

	return fmt.Sprintf("Network{layers=%d, architecture=%s}", totalLayers, architecture)
}
