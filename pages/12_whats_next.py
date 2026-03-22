import streamlit as st

st.title("What's Next?")
st.write("Congratulations! You've explored the core concepts of deep learning.")

st.subheader("What You've Learned")
st.write(
    "- **The Neuron:** How a single perceptron computes a weighted sum and applies an activation\n"
    "- **Activation Functions:** Why non-linearity is essential and how different activations compare\n"
    "- **Network Architecture:** How layers and neurons are organized, and how to count parameters\n"
    "- **Forward Pass:** How data flows through layers via matrix multiplication + activation\n"
    "- **Loss Functions:** How MSE and Cross-Entropy measure prediction error\n"
    "- **Backpropagation:** How the chain rule computes gradients for every weight\n"
    "- **Training Loop:** The forward-loss-backward-update cycle that trains a network\n"
    "- **Optimizers:** Why Adam outperforms plain SGD and how momentum helps\n"
    "- **Overfitting:** How regularization (L2, Dropout, Early Stopping) prevents memorization"
)

st.divider()

st.subheader("From NumPy to Real Frameworks")
st.write(
    "Everything in this app was built with pure NumPy — no deep learning framework. "
    "That's great for understanding the math, but real projects use frameworks that handle "
    "the details automatically:"
)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("#### PyTorch")
    st.write(
        "The most popular framework for research. "
        "Dynamic computation graphs, intuitive Python API."
    )
    st.write("[pytorch.org](https://pytorch.org)")
with col2:
    st.markdown("#### TensorFlow / Keras")
    st.write(
        "Great for production deployment. "
        "Keras provides a high-level API that's beginner-friendly."
    )
    st.write("[tensorflow.org](https://tensorflow.org)")
with col3:
    st.markdown("#### fast.ai")
    st.write(
        "Built on PyTorch with sensible defaults. "
        "Great for getting started quickly."
    )
    st.write("[fast.ai](https://www.fast.ai)")

st.divider()

st.subheader("Topics to Explore Next")

with st.expander("Convolutional Neural Networks (CNNs)"):
    st.write(
        "CNNs are designed for image data. Instead of connecting every neuron to every input, "
        "they use small **filters** that slide across the image, detecting patterns like edges, "
        "textures, and shapes. Key concepts: convolution, pooling, feature maps."
    )

with st.expander("Recurrent Neural Networks (RNNs) & LSTMs"):
    st.write(
        "RNNs process **sequential data** (text, time series) by maintaining a hidden state "
        "that gets updated at each time step. LSTMs and GRUs solve the vanishing gradient "
        "problem in long sequences."
    )

with st.expander("Transformers & Attention"):
    st.write(
        "Transformers replaced RNNs for most language tasks. The key innovation is **self-attention**, "
        "which lets the model look at all positions in the input simultaneously. "
        "GPT, BERT, and all modern LLMs are based on the Transformer architecture."
    )

with st.expander("Generative Models"):
    st.write(
        "**GANs** (Generative Adversarial Networks) pit two networks against each other to generate "
        "realistic images. **Diffusion models** (like Stable Diffusion) learn to gradually "
        "denoise random noise into structured outputs."
    )

st.divider()

st.subheader("Related Apps")
st.write(
    "- **ML No Code** — Learn traditional machine learning with an interactive, code-free app\n"
    "- **LLM Explainer** — Understand how Large Language Models work"
)

st.divider()
st.write("Built with Streamlit and pure NumPy. No GPU required.")
