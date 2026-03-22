# Deep Learning Explainer

This app was built to demonstrate core aspects of the  https://odsc.ai  Engineering Accelerator courses

Its interactive Streamlit app that teaches deep learning from scratch — one concept at a time, with live visualizations you can manipulate.

Built on a from-scratch NumPy neural network engine. No PyTorch, no TensorFlow — just the raw math so you can see exactly what's happening.

## What You'll Learn

| Step | Page | What It Covers |
|------|------|---------------|
| 1 | The Neuron | A single neuron drawing a decision boundary. Adjust weights and bias by hand. |
| 2 | Activation Functions | Compare ReLU, Sigmoid, Tanh — see why non-linearity matters. |
| 3 | Build a Network | Design an architecture and watch the parameter count change. |
| 4 | Forward Pass | Step through a network layer by layer, seeing every multiplication. |
| 5 | Loss Functions | Drag a prediction slider and watch MSE and Cross-Entropy respond. |
| 6 | Backpropagation | See gradients flow backwards through the network. |
| 7 | Training Loop | Train a network on 2D data and watch the decision boundary evolve in real time. |
| 8 | Optimizers | Same network, different optimizers — compare SGD, Momentum, Adam, RMSprop. |
| 9 | Overfitting | Toggle L2, Dropout, and Early Stopping to close the train/val gap. |
| 10 | Digit Recognition | Train a network to recognize handwritten digits — everything comes together. |

## Run Locally

```bash
git clone https://github.com/sheamusmcg/dl-explainer.git
cd dl-explainer
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Tech Stack

- **Streamlit** — interactive UI
- **NumPy** — neural network engine built from scratch (forward pass, backprop, optimizers)
- **Plotly** — interactive charts (loss curves, decision boundaries, activation functions)
- **Matplotlib** — network architecture diagrams
- **scikit-learn** — toy datasets and the 8×8 digits dataset

## Who This Is For

Anyone who wants to understand what's actually happening inside a neural network — students, self-taught developers, data scientists moving into deep learning, or anyone tired of black-box explanations.

No prior deep learning knowledge required. Basic familiarity with Python and high school math is enough.

---

Want me to commit this to the repo?
