package main

import (
	"fmt"

	"github.com/herlon214/go-neural/activation"
	"github.com/herlon214/go-neural/network"
	"github.com/herlon214/go-neural/tensor"
)

func main() {
	learningRate := tensor.Tensor(0.1)

	// Your XOR dataset
	dataset := []network.Sample{
		{Inputs: tensor.Tensors{0, 0}, Target: tensor.Tensors{0}},
		{Inputs: tensor.Tensors{0, 1}, Target: tensor.Tensors{1}},
		{Inputs: tensor.Tensors{1, 0}, Target: tensor.Tensors{1}},
		{Inputs: tensor.Tensors{1, 1}, Target: tensor.Tensors{0}},
	}

	hiddenLayers := []*network.LayerBuilder{
		network.NewLayerBuilder().
			SetNeurons(3).
			SetActivation(activation.NewReLU()),
	}
	outputLayer := network.NewLayerBuilder().SetNeurons(1).SetActivation(activation.NewSigmoid())

	neuralNetwork := network.New(
		2,
		hiddenLayers,
		outputLayer,
		learningRate,
	)

	neuralNetwork.Train(10_000, dataset)
	fmt.Println(neuralNetwork.String())
}
