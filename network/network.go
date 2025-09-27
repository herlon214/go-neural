package network

import (
	"fmt"
	"slices"

	"github.com/herlon214/go-neural/tensor"
)

type Loss tensor.Tensors
type Result tensor.Tensors

type Network struct {
	hiddenLayers []*Layer
	outputLayer  *Layer

	disableBackpropagation bool
}

func New(
	inputSize int,
	hiddenLayersBuilder []*LayerBuilder,
	outputLayerBuilder *LayerBuilder,
	learningRate tensor.Tensor,
) *Network {
	// Hidden layer
	hiddenLayers := make(Layers, 0, len(hiddenLayersBuilder))
	for i, builder := range hiddenLayersBuilder {
		// Next layer uses the previous layer size
		if i > 0 {
			inputSize = hiddenLayers.Last().Size()
		}

		hiddenLayers = append(hiddenLayers, builder.Build(inputSize, learningRate))
	}

	// Output layer
	outputLayerInputSize := hiddenLayers.Last().Size()

	return &Network{
		hiddenLayers: hiddenLayers,
		outputLayer:  outputLayerBuilder.Build(outputLayerInputSize, learningRate),
	}
}

func (n *Network) FeedForward(sample Sample) (Result, Loss) {
	// Feed hidden layers
	outputs := make([]tensor.Tensors, 0, len(n.hiddenLayers))
	inputs := make([]tensor.Tensors, 0, len(n.hiddenLayers)+1)
	inputs = append(inputs, sample.Inputs)

	for i, layer := range n.hiddenLayers {
		output := layer.FeedForward(inputs[i])
		outputs = append(outputs, output)
		inputs = append(inputs, output)
	}

	// Feed outputs layer
	finalOutput := n.outputLayer.FeedForward(inputs[len(inputs)-1])

	// Calculate loss
	loss := sample.Target.Clone().Subtract(finalOutput)

	// Back propagation on the output layer
	if !n.disableBackpropagation {
		outputErr := n.outputLayer.BackPropagation(loss, inputs[len(inputs)-1], finalOutput)

		for i, layer := range slices.Backward(n.hiddenLayers) {
			outputErr = layer.BackPropagation(outputErr, inputs[i], outputs[i])
		}
	}

	return Result(finalOutput), Loss(loss)
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

func (n *Network) DisableBackpropagation() {
	n.disableBackpropagation = true
}

func (n *Network) EnableBackpropagation() {
	n.disableBackpropagation = false
}
