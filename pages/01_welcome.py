import streamlit as st
from components.ui_helpers import next_step_button
from components import explanations

st.title("Deep Learning Explainer")
st.write("**Understand Deep Learning, One Layer at a Time**")

# Visual roadmap
st.subheader("Your Learning Journey")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown("### 1. Basics")
    st.write("Neuron, Activations, Architecture")
with col2:
    st.markdown("### 2. Inside")
    st.write("Forward Pass")
with col3:
    st.markdown("### 3. Learning")
    st.write("Loss, Backprop, Training")
with col4:
    st.markdown("### 4. Advanced")
    st.write("Optimizers, Regularization")
with col5:
    st.markdown("### 5. Next")
    st.write("Frameworks, CNNs, Transformers")

st.divider()

# Deep Learning vs ML
st.subheader("Deep Learning vs Traditional ML")
col_ml, col_dl = st.columns(2)
with col_ml:
    st.markdown("#### Traditional ML")
    st.write(
        "- You choose the features\n"
        "- Simpler models (trees, linear)\n"
        "- Small, structured datasets\n"
        "- Easy to interpret"
    )
with col_dl:
    st.markdown("#### Deep Learning")
    st.write(
        "- Network learns its own features\n"
        "- Layers of neurons\n"
        "- Images, text, audio, large data\n"
        "- Harder to interpret, more powerful"
    )

with st.expander("Learn more: Deep Learning vs Traditional ML"):
    st.markdown(explanations.WHAT_IS_DEEP_LEARNING)

# Key terms — concise definitions
st.subheader("Key Terms")
st.markdown(
    """
| Term | Definition |
|------|-----------|
| **Neuron** | Takes inputs, multiplies by weights, adds bias, applies activation. |
| **Layer** | Group of neurons at one stage. Input -> Hidden -> Output. |
| **Weight** | How much each input matters. Learned during training. |
| **Bias** | Shifts the output. Like an intercept. |
| **Activation** | Non-linear function after the weighted sum. Gives the network its power. |
| **Loss** | How wrong the prediction is. Lower = better. |
| **Gradient** | Direction to adjust weights to reduce loss. |
"""
)

next_step_button("pages/02_the_neuron.py", "Next: The Neuron")
