package network

import (
	"fmt"
	"math/rand"

	"github.com/herlon214/go-neural/tensor"
)

type Sample struct {
	Inputs tensor.Tensors
	Target tensor.Tensors
}

type Samples []Sample

func (s Samples) Clone() Samples {
	cloned := make([]Sample, len(s))
	copy(cloned, s)

	return cloned
}

func (s Samples) Shuffle() Samples {
	rand.Shuffle(len(s), func(i, j int) { s[i], s[j] = s[j], s[i] })

	return s
}

func (n *Network) Train(epochs int, samples Samples) {
	for i := range epochs {
		current := samples.Clone().Shuffle()

		for _, sample := range current {
			result, loss := n.FeedForward(sample)

			fmt.Printf("#%d -> %v = %v (loss = %v)\n", i, sample.Inputs, result, loss)
		}
	}
}
