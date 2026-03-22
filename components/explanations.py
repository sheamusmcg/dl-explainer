"""Long-form educational markdown content for expanders."""

WHAT_IS_DEEP_LEARNING = """
### Deep Learning vs Traditional Machine Learning

**Traditional ML** (like decision trees or logistic regression) relies on humans to pick the
right features. You might manually create features like "average word length" or "number of
edges" and feed those into a model.

**Deep learning** lets the network *discover its own features*. The early layers learn simple
patterns (edges, curves), and deeper layers combine them into complex concepts (faces, words,
meaning).

**Why "deep"?** It refers to the many *layers* stacked on top of each other. A network with
1 hidden layer is "shallow." A network with many hidden layers is "deep."

**When to use deep learning:**
- You have a lot of data
- The patterns are complex (images, text, audio)
- You don't know which features matter

**When traditional ML is better:**
- Small datasets (deep learning overfits easily)
- Tabular data with clear features
- You need interpretability
"""

WHAT_IS_A_NEURON = """
### The Artificial Neuron (Perceptron)

An artificial neuron is inspired by biological neurons, but much simpler. It does three things:

1. **Multiply** each input by a weight (how important is this input?)
2. **Sum** all the weighted inputs plus a bias term
3. **Apply** an activation function to produce the output

Mathematically: **y = activation(w1*x1 + w2*x2 + ... + b)**

The **weights** control how much each input matters. The **bias** shifts the decision
boundary, like an intercept in a line equation.

A single neuron can only draw a *straight line* to separate data. That's why we need
networks of many neurons to solve complex problems.
"""

ACTIVATION_INTUITION = """
### Why Do We Need Activation Functions?

Without activation functions, a neural network is just a series of linear transformations.
No matter how many layers you stack, the result is still a straight line (or hyperplane).

**Non-linearity** is what gives neural networks their power. Activation functions introduce
curves and bends, allowing the network to learn complex patterns like spirals, clusters,
and irregular shapes.

**Common activations:**
- **ReLU** (Rectified Linear Unit): Simple, fast, works well in practice. Outputs 0 for
  negative inputs, passes positive inputs through unchanged.
- **Sigmoid**: Squashes values to (0, 1). Good for probabilities, but can cause vanishing
  gradients in deep networks.
- **Tanh**: Like sigmoid but outputs (-1, 1). Zero-centered, which helps training.
- **Leaky ReLU**: Like ReLU but allows a small gradient for negative inputs, avoiding
  "dead neurons."
"""

NETWORK_DEPTH = """
### How Deep Is Deep Enough?

There's no magic formula, but here are some guidelines:

- **1 hidden layer** can approximate any continuous function (Universal Approximation
  Theorem), but may need very many neurons.
- **2-3 hidden layers** are sufficient for most practical problems.
- **Very deep networks** (10+ layers) are used for images (CNNs) and text (Transformers),
  but require special techniques (batch normalization, skip connections).

**More neurons per layer** = wider network = more parameters = more capacity.
**More layers** = deeper network = can learn more abstract features.

The tradeoff: bigger networks are more powerful but need more data and are more likely
to overfit.
"""

FORWARD_PASS_EXPLANATION = """
### What Happens Inside a Neural Network?

The **forward pass** is how data flows through the network to produce a prediction:

1. Input values enter the first layer
2. Each neuron computes: **z = weights . inputs + bias** (the weighted sum)
3. The activation function is applied: **a = activation(z)**
4. These activations become the inputs for the next layer
5. Repeat until the output layer

This is just matrix multiplication + activation, repeated at each layer. The entire
forward pass can be written as:

**Layer 1:** a1 = activation(W1 . x + b1)
**Layer 2:** a2 = activation(W2 . a1 + b2)
**Output:** y = activation(W3 . a2 + b3)
"""

LOSS_INTUITION = """
### Why Do We Need a Loss Function?

The loss function measures **how wrong** the model's prediction is. It gives us a single
number: lower is better.

Without a loss function, we have no way to tell the network how to improve. The loss
function is the network's "report card."

**For regression** (predicting numbers): Use **Mean Squared Error (MSE)**. It penalizes
large errors more than small ones (because of the squaring).

**For classification** (predicting categories): Use **Cross-Entropy**. It measures how
different the predicted probability distribution is from the true distribution. It
heavily penalizes confident wrong predictions.

The goal of training is to find the weights that **minimize** the loss function.
"""

BACKPROP_INTUITION = """
### Backpropagation: The Chain Rule in Action

Backpropagation answers the question: **"How much did each weight contribute to the error?"**

It uses the **chain rule** from calculus to work backwards:

1. Start with the loss (how wrong was the prediction?)
2. How much did the output layer's weights contribute to that loss?
3. How much did the hidden layer's activations contribute to the output?
4. How much did the hidden layer's weights contribute to those activations?

At each step, we multiply the gradients together (chain rule). This tells us the
**direction** and **magnitude** to adjust each weight.

**Key insight:** Gradients flow backwards through the network. Layers closer to the
output get larger gradients (they adjust more). Layers further back can get very small
gradients — this is the **vanishing gradient problem**.
"""

TRAINING_LOOP_EXPLANATION = """
### The Training Loop

Training a neural network is an iterative process:

1. **Forward pass:** Feed data through the network, get predictions
2. **Compute loss:** Compare predictions to true labels
3. **Backward pass:** Compute gradients (how to adjust each weight)
4. **Update weights:** Nudge weights in the direction that reduces loss
5. **Repeat** for many epochs

An **epoch** is one complete pass through the entire training dataset.

The **learning rate** controls how big each weight update is:
- Too high: the network overshoots and oscillates
- Too low: training is very slow
- Just right: smooth convergence to a good solution
"""

OPTIMIZER_INTUITION = """
### Why Do We Need Different Optimizers?

**Plain SGD** (Stochastic Gradient Descent) updates weights using only the current
gradient. It works, but has problems:
- Gets stuck in flat regions (small gradients = tiny updates)
- Oscillates in steep, narrow valleys
- Same learning rate for all parameters

**Momentum** adds a "velocity" term — like a ball rolling downhill, it accumulates
speed in consistent directions and dampens oscillations.

**Adam** (Adaptive Moment Estimation) adapts the learning rate *per parameter* based
on the history of gradients. Parameters that rarely get large gradients get larger
updates. It combines the benefits of Momentum and RMSprop.

**RMSprop** divides the learning rate by a running average of gradient magnitudes,
preventing any single parameter from dominating.

In practice, **Adam** is the most popular default choice.
"""

OVERFITTING_INTUITION = """
### Overfitting: Memorization vs Learning

**Overfitting** happens when a model memorizes the training data (including its noise)
instead of learning the underlying pattern. It performs great on training data but
poorly on new data.

**Signs of overfitting:**
- Training loss keeps decreasing, but validation loss starts increasing
- The decision boundary is overly complex, fitting around individual points

**How to fight overfitting:**

- **L2 Regularization:** Adds a penalty for large weights, forcing the network to
  use smaller, more distributed weights. This makes the decision boundary smoother.

- **Dropout:** Randomly turns off neurons during training. This prevents the network
  from relying too heavily on any single neuron, encouraging redundancy.

- **Early Stopping:** Monitor validation loss and stop training when it starts
  increasing, even if training loss is still decreasing.

- **More data:** The best cure for overfitting is more training data.
"""
