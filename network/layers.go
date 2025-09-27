package network

type Layers []*Layer

func (l Layers) Last() *Layer {
	return l[len(l)-1]
}
