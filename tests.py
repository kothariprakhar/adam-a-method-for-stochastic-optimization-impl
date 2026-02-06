import unittest
import torch
import torch.nn as nn
import math

# Assuming AdamFromScratch is defined in the file or imported
# Copying the class definition here for self-contained testing context if needed,
# but in a real scenario, we import it. 
# For the purpose of this output, we assume the class provided in the prompt is available.

class TestAdamFromScratch(unittest.TestCase):
    def setUp(self):
        self.lr = 0.001
        self.betas = (0.9, 0.999)
        self.eps = 1e-8

    def test_initialization_checks(self):
        """Test that invalid hyperparameters raise errors."""
        param = torch.tensor([1.0])
        with self.assertRaises(ValueError):
            AdamFromScratch([param], lr=-0.1)
        with self.assertRaises(ValueError):
            AdamFromScratch([param], eps=-1e-8)
        with self.assertRaises(ValueError):
            AdamFromScratch([param], betas=(1.1, 0.999))

    def test_single_step_logic(self):
        """Verify the mathematical update of a single step against manual calculation."""
        # Setup a simple parameter
        param = torch.tensor([1.0], requires_grad=True)
        optimizer = AdamFromScratch([param], lr=self.lr, betas=self.betas, eps=self.eps)

        # Define a gradient manually
        param.grad = torch.tensor([0.1])
        
        # Manual Calculation of what should happen
        # t = 1
        g = 0.1
        beta1, beta2 = self.betas
        
        # m_1 = beta1 * m_0 + (1-beta1) * g = 0.9 * 0 + 0.1 * 0.1 = 0.01
        m = (beta1 * 0) + (1 - beta1) * g
        # v_1 = beta2 * v_0 + (1-beta2) * g^2 = 0.999 * 0 + 0.001 * 0.01 = 0.00001
        v = (beta2 * 0) + (1 - beta2) * (g ** 2)
        
        # Bias correction
        m_hat = m / (1 - beta1 ** 1)
        v_hat = v / (1 - beta2 ** 1)
        
        # Update
        denom = math.sqrt(v_hat) + self.eps
        expected_param_value = 1.0 - self.lr * (m_hat / denom)

        # Perform Step
        optimizer.step()

        self.assertTrue(torch.isclose(param, torch.tensor([expected_param_value]), atol=1e-7).item(),
                        f"Update failed. Expected {expected_param_value}, got {param.item()}")

    def test_convergence_quadratic(self):
        """Test if the optimizer can minimize a simple quadratic function y = x^2."""
        param = torch.tensor([10.0], requires_grad=True)
        optimizer = AdamFromScratch([param], lr=0.1)

        # Optimization loop
        for _ in range(200):
            optimizer.zero_grad()
            loss = param ** 2
            loss.backward()
            optimizer.step()

        # Should be close to 0
        self.assertTrue(torch.abs(param).item() < 0.1,
                        f"Optimizer failed to converge on simple quadratic. Final value: {param.item()}")

    def test_weight_decay(self):
        """Test that weight decay is applied to gradients correctly."""
        param = torch.tensor([1.0], requires_grad=True)
        wd = 0.1
        optimizer = AdamFromScratch([param], lr=0.1, weight_decay=wd)

        # Gradient g = 1.0
        param.grad = torch.tensor([1.0])
        
        # Save original data for manual verification references
        # If WD is applied, effective gradient = grad + wd * param = 1.0 + 0.1 * 1.0 = 1.1
        
        # We just check if the internal state reflects the higher gradient magnitude indirectly 
        # or simpler: check the param update magnitude is larger than without WD.
        
        # Step with WD
        optimizer.step()
        param_with_wd = param.item()
        
        # Step without WD (reset param)
        param2 = torch.tensor([1.0], requires_grad=True)
        optimizer2 = AdamFromScratch([param2], lr=0.1, weight_decay=0.0)
        param2.grad = torch.tensor([1.0])
        optimizer2.step()
        param_no_wd = param2.item()

        # With L2 regularization (weight decay), the parameter should decay faster/move more towards 0 
        # given the gradient direction and param sign are positive.
        # Original: 1.0. Gradient: 1.0. Step moves negative.
        # WD adds to gradient -> larger gradient -> larger step -> smaller resulting parameter value.
        self.assertLess(param_with_wd, param_no_wd,
                        "Weight decay did not result in a larger step magnitude as expected.")

if __name__ == '__main__':
    unittest.main()