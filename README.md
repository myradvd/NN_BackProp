## NNbackprop-first-principles

This project is an independent implementation of a scalar-valued backpropagation engine, built off first principles from Andrej Karpathy's MicroGrad. It includes the re-derivation of automatic differentiation from a calculus and graph-theoretic foundation, with detailed explanations.

It extends MicroGrad's idea and scope to mathematically explain and visualise concepts like vanishing gradients, and the effect of the learning rate on gradient descent.

### Why I built it this way

This is a personal learning project. I wanted to understand what actually happens when a neural network learns, not just call `loss.backward()` and trust it.

So I kept it as simple as I could. Everything is a single number (a scalar), not a matrix, so every value and every gradient can be printed, drawn and checked by hand. It's one notebook, with no framework and nothing hidden. If I can't work a step out on paper, it doesn't go in.

### What's in the notebook

Everything is in [Try_Out.ipynb](Try_Out.ipynb), in four parts.

**Part 1: Building the engine**
- A `Value` class that wraps a number and records how it was made (its inputs and the operation), so the whole calculation can be traced afterwards.
- This first version is forward-only, with no automatic gradients, so Part 2 has to do backprop by hand.
- Graphviz drawings of the computation graph, showing each value's data and gradient.
- A first example neuron: `σ(x1·w1 + x2·w2 + b)`.
- Why we need activation functions: a straight line can separate simple data, but curved boundaries need non-linearity, and ReLU's "bends" build those curves.

**Part 2: Backpropagation by hand**
- What a derivative means here, from its limit definition, and why the output's gradient with respect to itself is 1.
- Why gradients flow right to left, from the output back to the inputs.
- The sigmoid derivative `σ(1−σ)`, worked through on the example (it comes out tiny, ~0.00055).
- The vanishing gradient problem:
  - proof that the sigmoid derivative can never be more than 0.25
  - what that does to weight updates
  - fixes: other activations such as ReLU (and its "dying" problem), and normalising inputs
- The chain rule in plain language.
- A tanh neuron, backpropagated step by step by setting each `.grad` by hand: through the activation, then the bias, then the addition nodes, with a short proof of why addition passes gradients through unchanged.

**Part 3: Automatic backpropagation**
- A second version of `Value` where every operation stores its own backward rule, plus a `backward()` that sorts the graph (topological sort) and runs those rules from the output back.
- Neurons, layers and multi-layer perceptrons (MLPs), explained with a simplified Titanic survival example (age, gender, class as inputs).
- Common activation functions: ReLU, sigmoid, softmax, tanh.

**Part 4: Learning with gradient descent** *(in progress)*
- How errors drive learning: why we step *against* the gradient, why least squares fits data but doesn't separate classes, and a worked softmax example (cat / dog / fish scores turned into probabilities), leading into cross-entropy.
- Minimising a loss `L = (w − 3)²` with the engine: computing the gradient, then making the update `w_new = w − η·∂L/∂w` by hand.
- Plots of:
  - gradient descent stepping down the loss curve to its minimum
  - how the gradient `2(w − 3)` changes across the curve
  - how learning rates of 0.01, 0.1 and 0.8 change how fast the loss falls

### What I added on top of micrograd

micrograd is a tiny autograd engine (~100 lines) plus a small neural-net library. These parts are taken from it as-is: the `Module`, `Neuron`, `Layer` and `MLP` classes (credited in the notebook), and the core idea of a `Value` with a `_backward` function and a topological-sort `backward()`. The graph-drawing code follows Karpathy's lecture.

On top of that, I added:

**To the engine**
- **A forward-only version first.** The engine is built in two stages, so the gradients can be worked out by hand before they're automated.
- **Subtraction and division with their own derivatives.** micrograd builds `a − b` as `a + (−b)` and `a / b` as `a · b⁻¹`. Here each has its own backward rule (`∂(a−b)/∂b = −1`, `∂(a/b)/∂b = −a/b²`) written out in the comments.
- **Powers where the exponent is also a `Value`.** micrograd only allows a constant exponent. Here `a ** b` handles both, using `∂(aᵇ)/∂b = ln(a)·aᵇ`.
- **tanh and sigmoid as built-in operations,** each with its derivative, alongside ReLU.
- **Labels on values,** so every node in the graph drawings is named (`x1`, `w1`, `bias`, ...).

**Explanations and experiments**
- Backprop worked out by hand on a real neuron before using the automatic version.
- The maths of the vanishing gradient problem, not just its name: the 0.25 bound on the sigmoid derivative and what it does over many layers.
- Gradient descent and the learning rate, shown with plots instead of only described.
- My own intuition for activation functions, loss functions, softmax and cross-entropy, from a classification point of view.

### Still to do

Part 4 is unfinished. I still want to:
- compare SGD with Adam and other optimisers
- add cross-entropy as a loss and use it to train a classifier
- train a model on real data (e.g. the Titanic dataset used as the example)

### Running it

```
pip install -r requirements.txt
```

Note: To run the Try_Out ipynb file on VsCode, you need to have GraphViz installed (installers can be found here: https://graphviz.org/download/):

- **Windows:** the installer from the link above
- **Mac:** `brew install graphviz`
- **Linux:** `sudo apt install graphviz` (or your distro's package manager)

The first code cell adds the default install location for your OS to PATH and prints where it found Graphviz. If it prints `None`, add your install folder to `graphviz_paths` in that cell, to ensure you don't run into FileNotFound Errors.
