# adversarial_example_generator

A Python library to generate adversarial examples for classification models

Use the notebook in the `examples` folder to see how to use the library. The notebook will automatically install the necessary libraries.

## Possible improvements

- [ ] Add the possibility of specifying a target probability instead of a fixed number of iterations, and the model keeps iteratively adding noise until the model predicts the target class with at least the specified probability. The user should also specify a max number of iterations to avoid infinite loops.
