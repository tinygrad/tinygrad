from tinygrad import Tensor, nn
from tinygrad.nn.optim import SGD

import unittest

# Through experimental testing, I've found that if any of these numbers go any lower, the bug disappears
INPUT_DIMS = 868
HIDDEN_DIMS = 30
OUTPUT_DIMS = 220

class TestBug(unittest.TestCase):
    def optimal_mse(self, y_pred: Tensor, y_true: Tensor, weights: Tensor) -> Tensor:
        """
        This is sort of the core of what I was trying to do. I'm trying to avoid overfitting for 
        ARC-AGI and I figured one way to do it might be to add a penalty if the weights are above 0
        To encourage 0 weights that basically don't do anything for the network's calculations
        Thus using the fewest possible weights.
        """
        mse = (y_pred - y_true).pow(2).mean()

        nonzero_penalty = weights.abs().sum() / (weights.shape[0]) # If this is removed, the bug disappears.

        return mse + nonzero_penalty # Add it so if it is smaller, loss is smaller

    class Model:
        def __init__(self):
            self.l3 = nn.Linear(INPUT_DIMS, HIDDEN_DIMS)
            self.l4 = nn.Linear(HIDDEN_DIMS, OUTPUT_DIMS)

        def __call__(self, x:Tensor) -> Tensor:
            x = self.l3(x)
            return self.l4(x)

    def test_bug(self):
        net = self.Model()

        opt = SGD([net.l3.weight, net.l4.weight], lr=3e-4)


        with Tensor.train():
            input = Tensor.zeros(INPUT_DIMS)
            output = Tensor.zeros(OUTPUT_DIMS)

            opt.zero_grad()
            output_pred = net(input)
            flat_l3 = net.l3.weight.reshape(INPUT_DIMS * HIDDEN_DIMS)
            flat_l4 = net.l4.weight.reshape(HIDDEN_DIMS * OUTPUT_DIMS)
            loss = self.optimal_mse(output_pred, output, flat_l3.cat(flat_l4))
            loss.backward()
            opt.step()
            try:
                print(loss.numpy())

            except Exception as e:
                self.fail(f"Strange bug detected: {e}")


if __name__ == '__main__':
    unittest.main()
