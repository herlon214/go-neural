package tensor

import (
	"math"
	"testing"
)

func TestTensor_Float64(t *testing.T) {
	tests := []struct {
		name     string
		tensor   Tensor
		expected float64
	}{
		{"positive value", Tensor(3.14), 3.14},
		{"negative value", Tensor(-2.5), -2.5},
		{"zero", Tensor(0), 0},
		{"large value", Tensor(1000000.5), 1000000.5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.tensor.Float64()
			if result != tt.expected {
				t.Errorf("Tensor.Float64() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestTensors_Multiply(t *testing.T) {
	tests := []struct {
		name     string
		tensors  Tensors
		expected Tensor
	}{
		{
			name:     "multiply positive numbers",
			tensors:  Tensors{2.0, 3.0, 4.0},
			expected: Tensor(24.0),
		},
		{
			name:     "multiply with negative",
			tensors:  Tensors{2.0, -3.0, 4.0},
			expected: Tensor(-24.0),
		},
		{
			name:     "multiply with zero",
			tensors:  Tensors{2.0, 0.0, 4.0},
			expected: Tensor(0.0),
		},
		{
			name:     "single tensor",
			tensors:  Tensors{5.0},
			expected: Tensor(5.0),
		},
		{
			name:     "multiply fractions",
			tensors:  Tensors{0.5, 0.2, 10.0},
			expected: Tensor(1.0),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.tensors.Multiply()
			if math.Abs(float64(result-tt.expected)) > 1e-10 {
				t.Errorf("Tensors.Multiply() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestTensors_Sum(t *testing.T) {
	tests := []struct {
		name     string
		receiver Tensors
		others   Tensors
		expected Tensors
	}{
		{
			name:     "sum positive numbers",
			receiver: Tensors{1.0, 2.0, 3.0},
			others:   Tensors{4.0, 5.0, 6.0},
			expected: Tensors{5.0, 7.0, 9.0},
		},
		{
			name:     "sum with negatives",
			receiver: Tensors{1.0, -2.0, 3.0},
			others:   Tensors{-1.0, 2.0, -3.0},
			expected: Tensors{0.0, 0.0, 0.0},
		},
		{
			name:     "sum with zeros",
			receiver: Tensors{1.0, 2.0, 3.0},
			others:   Tensors{0.0, 0.0, 0.0},
			expected: Tensors{1.0, 2.0, 3.0},
		},
		{
			name:     "sum fractions",
			receiver: Tensors{0.1, 0.2, 0.3},
			others:   Tensors{0.4, 0.5, 0.6},
			expected: Tensors{0.5, 0.7, 0.9},
		},
		{
			name:     "single element",
			receiver: Tensors{10.0},
			others:   Tensors{5.0},
			expected: Tensors{15.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Make a copy to preserve original for comparison
			original := make(Tensors, len(tt.receiver))
			copy(original, tt.receiver)

			// Call Sum method (modifies receiver in-place)
			tt.receiver.Sum(tt.others)

			// Verify the result
			if len(tt.receiver) != len(tt.expected) {
				t.Errorf("Sum() result length = %v, want %v", len(tt.receiver), len(tt.expected))
				return
			}

			for i := range tt.receiver {
				if math.Abs(float64(tt.receiver[i]-tt.expected[i])) > 1e-10 {
					t.Errorf("Sum() result[%d] = %v, want %v", i, tt.receiver[i], tt.expected[i])
				}
			}

			// Verify that the original was modified (in-place operation)
			for i := range original {
				if math.Abs(float64(original[i]+tt.others[i]-tt.receiver[i])) > 1e-10 {
					t.Errorf("Sum() should modify receiver in-place. Expected original[%d] + others[%d] = receiver[%d]", i, i, i)
				}
			}
		})
	}
}

func TestTensors_Sum_EdgeCases(t *testing.T) {
	t.Run("empty tensors", func(t *testing.T) {
		receiver := Tensors{}
		others := Tensors{}
		receiver.Sum(others)

		if len(receiver) != 0 {
			t.Errorf("Sum() with empty tensors should remain empty, got length %d", len(receiver))
		}
	})

	t.Run("mismatched lengths should not panic", func(t *testing.T) {
		defer func() {
			if r := recover(); r != nil {
				t.Errorf("Sum() with mismatched lengths should not panic, but it did: %v", r)
			}
		}()

		receiver := Tensors{1.0, 2.0, 3.0}
		others := Tensors{1.0, 2.0} // shorter

		// This should only modify the first len(others) elements
		receiver.Sum(others)

		// Verify first two elements were modified, third unchanged
		expected := Tensors{2.0, 4.0, 3.0}
		for i := range expected {
			if math.Abs(float64(receiver[i]-expected[i])) > 1e-10 {
				t.Errorf("Sum() result[%d] = %v, want %v", i, receiver[i], expected[i])
			}
		}
	})
}

// Benchmark tests to verify performance
func BenchmarkTensors_Multiply(b *testing.B) {
	tensors := Tensors{2.0, 3.0, 4.0, 5.0, 6.0}

	for b.Loop() {
		_ = tensors.Multiply()
	}
}

func BenchmarkTensors_Sum(b *testing.B) {
	receiver := Tensors{1.0, 2.0, 3.0, 4.0, 5.0}
	others := Tensors{1.0, 1.0, 1.0, 1.0, 1.0}

	for b.Loop() {
		// Reset receiver for each iteration
		copy(receiver, Tensors{1.0, 2.0, 3.0, 4.0, 5.0})
		receiver.Sum(others)
	}
}

func TestTensors_Subtract(t *testing.T) {
	tests := []struct {
		name     string
		receiver Tensors
		others   Tensors
		expected Tensors
	}{
		{
			name:     "subtract positive numbers",
			receiver: Tensors{5.0, 7.0, 9.0},
			others:   Tensors{1.0, 2.0, 3.0},
			expected: Tensors{4.0, 5.0, 6.0},
		},
		{
			name:     "subtract with negatives",
			receiver: Tensors{1.0, -2.0, 3.0},
			others:   Tensors{-1.0, 2.0, -3.0},
			expected: Tensors{2.0, -4.0, 6.0},
		},
		{
			name:     "subtract with zeros",
			receiver: Tensors{1.0, 2.0, 3.0},
			others:   Tensors{0.0, 0.0, 0.0},
			expected: Tensors{1.0, 2.0, 3.0},
		},
		{
			name:     "subtract resulting in negatives",
			receiver: Tensors{1.0, 2.0, 3.0},
			others:   Tensors{2.0, 4.0, 6.0},
			expected: Tensors{-1.0, -2.0, -3.0},
		},
		{
			name:     "subtract fractions",
			receiver: Tensors{0.9, 0.7, 0.5},
			others:   Tensors{0.4, 0.2, 0.1},
			expected: Tensors{0.5, 0.5, 0.4},
		},
		{
			name:     "single element",
			receiver: Tensors{10.0},
			others:   Tensors{3.0},
			expected: Tensors{7.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Make a copy to preserve original for comparison
			original := make(Tensors, len(tt.receiver))
			copy(original, tt.receiver)

			// Call Subtract method (modifies receiver in-place)
			tt.receiver.Subtract(tt.others)

			// Verify the result
			if len(tt.receiver) != len(tt.expected) {
				t.Errorf("Subtract() result length = %v, want %v", len(tt.receiver), len(tt.expected))
				return
			}

			for i := range tt.receiver {
				if math.Abs(float64(tt.receiver[i]-tt.expected[i])) > 1e-10 {
					t.Errorf("Subtract() result[%d] = %v, want %v", i, tt.receiver[i], tt.expected[i])
				}
			}

			// Verify that the original was modified (in-place operation)
			for i := range original {
				if math.Abs(float64(original[i]-tt.others[i]-tt.receiver[i])) > 1e-10 {
					t.Errorf("Subtract() should modify receiver in-place. Expected original[%d] - others[%d] = receiver[%d]", i, i, i)
				}
			}
		})
	}
}

func TestTensors_Subtract_EdgeCases(t *testing.T) {
	t.Run("empty tensors", func(t *testing.T) {
		receiver := Tensors{}
		others := Tensors{}
		receiver.Subtract(others)

		if len(receiver) != 0 {
			t.Errorf("Subtract() with empty tensors should remain empty, got length %d", len(receiver))
		}
	})

	t.Run("mismatched lengths should not panic", func(t *testing.T) {
		defer func() {
			if r := recover(); r != nil {
				t.Errorf("Subtract() with mismatched lengths should not panic, but it did: %v", r)
			}
		}()

		receiver := Tensors{5.0, 6.0, 7.0}
		others := Tensors{1.0, 2.0} // shorter

		// This should only modify the first len(others) elements
		receiver.Subtract(others)

		// Verify first two elements were modified, third unchanged
		expected := Tensors{4.0, 4.0, 7.0}
		for i := range expected {
			if math.Abs(float64(receiver[i]-expected[i])) > 1e-10 {
				t.Errorf("Subtract() result[%d] = %v, want %v", i, receiver[i], expected[i])
			}
		}
	})
}

func BenchmarkTensors_Subtract(b *testing.B) {
	receiver := Tensors{5.0, 6.0, 7.0, 8.0, 9.0}
	others := Tensors{1.0, 1.0, 1.0, 1.0, 1.0}

	for b.Loop() {
		// Reset receiver for each iteration
		copy(receiver, Tensors{5.0, 6.0, 7.0, 8.0, 9.0})
		receiver.Subtract(others)
	}
}
