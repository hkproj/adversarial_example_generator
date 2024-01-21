import torch
import torch.nn as nn

class AdversarialExampleGenerator:

    def __init__(self, model: nn.Module, epsilon: float = 0.25, alpha: float = 0.025, loss_fn: nn.Module = None) -> None:
        """ 
            Initializes the generator with a model, epsilon, alpha, and loss function.

            Args:
                model: The Torch model to generate adversarial examples for.
                epsilon: The amount of perturbation. Defaults to 0.25.
                alpha: The step size for the iterative method. Defaults to 0.025.
                loss_fn: The loss function to use when generating the adversarial example. Defaults to nn.CrossEntropyLoss().
        """

        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        if loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = loss_fn

    def generate(self, input: torch.Tensor, target_class: int, num_steps: int = 10) -> torch.Tensor:
        """
            Generates an adversarial example using an iterative method called Fast Gradient Sign Method (FGSM).

            Args:
                input: The input tensor to generate an adversarial example for.
                target_class: The index of the class to target.
                num_steps: The number of steps to take when generating the adversarial example. Defaults to 10.

            Returns:
                An adversarial example tensor.
        """

        if num_steps <= 0:
            raise ValueError("Number of steps must be greater than zero")
        
        # Make sure the model is in evaluation mode
        assert self.model.training == False, "Model should be in evaluation mode"
        # Create a copy of the input tensor so that we don't affect the original
        assert type(target_class) == int, "Target class must be an integer"

        x_adv_prev_step = torch.tensor(input.data, requires_grad=True)
        for _ in range(num_steps):
            output = self.model(x_adv_prev_step)

            # Since the output is a probability distribution, the target class must be within the bounds of the output
            _, num_classes = output.shape
            if target_class >= num_classes:
                raise ValueError("Target class must be within bounds of the model output")
            
            # Evaluate the loss function on the adversarial example
            target = torch.tensor([target_class], requires_grad=False)
            loss = self.loss_fn(output, target)
            # Backpropagate the loss to the input
            loss.backward()

            # The formula for the iterative method is x_{n+1} = x_{n} - alpha * gradient
            # and then we need to clip x_n+1 within the range (x - epsilon, x + epsilon) where x is the original input.
            x_adv_curr_step = x_adv_prev_step.data - self.alpha * torch.sign(x_adv_prev_step.grad.data)
            x_adv_curr_step = torch.clamp(x_adv_curr_step, input-self.epsilon, input+self.epsilon)
            x_adv_prev_step.data = x_adv_curr_step
        return x_adv_prev_step
