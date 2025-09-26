package main

import (
	"fmt"

	"github.com/herlon214/go-neural/network"
	"github.com/herlon214/go-neural/tensor"
)

const learningRate = 0.1

func main() {
	input := tensor.Tensors{
		8, 2,
	}

	inputSize := len(input)
	neuralNetwork := network.New(inputSize, []int{1}, 1, learningRate)
	fmt.Println(neuralNetwork.String())

	for range 10 {
		loss := neuralNetwork.FeedForward(input)
		fmt.Println(loss)

	}
}
