"""Centralized tooltip text for all pages."""

NEURON = {
    "weight_w1": "How much influence input x1 has on the output.",
    "weight_w2": "How much influence input x2 has on the output.",
    "bias": "Shifts the decision boundary. Like an intercept in a line equation.",
    "activation": "The function applied after the weighted sum. Adds non-linearity.",
    "dataset": "Choose a toy dataset to classify with the neuron.",
    "noise": "How much random scatter to add to the data points.",
    "seed": "Random seed for reproducible data generation.",
}

ACTIVATION = {
    "functions": "Select which activation functions to display.",
    "x_range": "The input range to plot over.",
    "show_derivative": "Overlay the derivative (gradient) of each function.",
}

NETWORK = {
    "num_layers": "Number of hidden layers between input and output.",
    "neurons": "Number of neurons in this hidden layer.",
    "activation": "Activation function for this layer.",
    "output_neurons": "1 for binary classification, more for multi-class.",
}

FORWARD_PASS = {
    "input_x1": "First input value to feed through the network.",
    "input_x2": "Second input value to feed through the network.",
    "step": "Which layer of the computation to display.",
}

LOSS = {
    "loss_function": "The function that measures how wrong the prediction is.",
    "predicted": "The model's predicted value (adjustable).",
    "true_label": "The actual correct answer.",
}

BACKPROP = {
    "input_x1": "First input value.",
    "input_x2": "Second input value.",
    "target": "The desired output value.",
}

TRAINING = {
    "dataset": "Choose a toy 2D dataset to train on.",
    "hidden_layers": "Number of hidden layers in the network.",
    "neurons_per_layer": "Neurons in each hidden layer.",
    "learning_rate": "Step size for weight updates. Too high = unstable, too low = slow.",
    "epochs": "Number of complete passes through the training data.",
    "activation": "Activation function for hidden layers.",
}

OPTIMIZERS = {
    "optimizers": "Select which optimizers to compare.",
    "learning_rate": "Step size for this optimizer.",
    "momentum": "How much of the previous update to carry forward (SGD only).",
    "beta1": "Exponential decay rate for first moment estimates (Adam).",
    "beta2": "Exponential decay rate for second moment estimates (Adam).",
}

OVERFITTING = {
    "noise": "More noise makes the problem harder and overfitting more likely.",
    "val_split": "Fraction of data reserved for validation (not used in training).",
    "complexity": "Network size: more neurons = more capacity to memorize.",
    "l2_lambda": "L2 regularization strength. Penalizes large weights.",
    "dropout_rate": "Fraction of neurons randomly turned off during training.",
    "early_stopping_patience": "Stop training if validation loss doesn't improve for this many epochs.",
}
