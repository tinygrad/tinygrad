import unittest
from tinygrad import Tensor
import numpy as np
import torch 


class TestMultipleGradient(unittest.TestCase):
    '''
    Set up an idealized system where the weights are initialized to the correct value
    of an all-ones tensor. Check that the gradient is zero.
    '''
    def total_loss(self, true_energy, model_energy, true_force, model_force):
        # Set up the energy-matching and force-matching objectives
        fm_loss = ((true_force - model_force)**2).mean() 
        em_loss = ((true_energy - model_energy)**2).mean()
        loss = fm_loss + em_loss
        # Predicted energy is exact, so d(loss)/d(weights) == 0
        return loss

    def test_pytorch_force_matching_loss_gradient(self):

        x_samples = torch.linspace(-10, 10, 100, requires_grad=True)
        weights = torch.ones_like(x_samples, requires_grad = True)
        true_energy = x_samples**2
        true_force = -2*x_samples

        model_energy = weights * x_samples**2

        # PyTorch doesn't allow you to run backward through the computation graph twice
        # unless you manually specify that it is okay through torch.autograd.grad()
        # The only reasonable equivalent in tinygrad is to call backward() twice, 
        # which does not throw an error as it would in PyTorch.
        model_force = -torch.autograd.grad(model_energy.sum(), x_samples, retain_graph=True)[0]

        loss = self.total_loss(true_energy=true_energy, model_energy=model_energy,
                               true_force=true_force, model_force=model_force)

        loss.backward()
        self.assertAlmostEqual(weights.grad.mean().item(), 0.,
                               msg="The pytorch weight gradient is not zero")
        
    def test_tinygrad_force_matching_loss_gradient(self):

        x_samples = Tensor(np.linspace(-10, 10, 100), requires_grad=True)
        weights = Tensor.ones_like(x_samples, requires_grad = True)
        true_energy = x_samples**2
        true_force = -2*x_samples

        model_energy = weights * x_samples**2

        model_energy.sum().backward()
        model_force = -x_samples.grad

        loss = self.total_loss(true_energy=true_energy, model_energy=model_energy,
                               true_force=true_force, model_force=model_force)

        loss.backward()
        self.assertAlmostEqual(weights.grad.mean().item(), 0.,
                               msg="The tinygrad weight gradient is not zero")

        
if __name__ == '__main__':
  unittest.main()
