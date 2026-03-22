import streamlit as st
import numpy as np
from components.ui_helpers import next_step_button
from components import tooltips, explanations
from components.viz_utils import plot_loss_function_curve

st.title("Loss Functions")

with st.expander("Where does the loss function fit in?", expanded=True):
    st.markdown(
        "After a forward pass, the network produces a **prediction**. But how do we know if it's any good? "
        "That's what the loss function answers.\n\n"
        "- It compares the network's prediction against the **true answer** (the label from your training data) "
        "and outputs a single number: the **loss**.\n"
        "- **High loss** = the prediction is far off. **Low loss** = the prediction is close.\n"
        "- The entire goal of training is to **minimize this number**. Backpropagation uses it to figure out "
        "which direction to nudge each weight.\n"
        "- Different problems need different loss functions: "
        "**MSE** for predicting numbers (regression), **BCE** for predicting yes/no probabilities (classification).\n\n"
        "Drag the slider below to simulate what happens as a prediction moves closer to or further from the true value."
    )

# Loss function selector
loss_name = st.selectbox(
    "Loss Function",
    ["mse", "binary_cross_entropy"],
    format_func=lambda x: "Mean Squared Error (MSE)" if x == "mse" else "Binary Cross-Entropy (BCE)",
    help=tooltips.LOSS["loss_function"],
)

st.divider()

if loss_name == "mse":
    st.subheader("Mean Squared Error")
    st.latex(r"MSE = \frac{1}{n} \sum (y_{pred} - y_{true})^2")

    col1, col2 = st.columns(2)
    with col1:
        y_true = st.number_input("True value", value=1.0, min_value=-5.0, max_value=5.0, step=0.1,
                                 help=tooltips.LOSS["true_label"])
    with col2:
        y_pred = st.slider("Predicted value", -5.0, 5.0, 0.0, 0.05,
                           help=tooltips.LOSS["predicted"])

    fig = plot_loss_function_curve("mse", y_true, pred_range=(-5.0, 5.0), current_pred=y_pred)
    st.plotly_chart(fig, use_container_width=True)

    loss_val = (y_pred - y_true) ** 2
    gradient = 2 * (y_pred - y_true)
    st.write(f"**Loss:** ({y_pred:.2f} - {y_true:.2f})^2 = **{loss_val:.4f}**")
    st.write(f"**Gradient:** 2 * ({y_pred:.2f} - {y_true:.2f}) = **{gradient:.4f}**")
    if gradient > 0:
        st.write("Gradient is positive -> the prediction should **decrease**.")
    elif gradient < 0:
        st.write("Gradient is negative -> the prediction should **increase**.")
    else:
        st.write("Gradient is zero -> the prediction is **perfect**!")

else:
    st.subheader("Binary Cross-Entropy")
    st.latex(r"BCE = -\frac{1}{n}\sum [y \cdot \log(\hat{y}) + (1-y) \cdot \log(1-\hat{y})]")

    col1, col2 = st.columns(2)
    with col1:
        y_true = st.selectbox("True label", [0, 1], index=1, help=tooltips.LOSS["true_label"])
    with col2:
        y_pred = st.slider("Predicted probability", 0.01, 0.99, 0.5, 0.01,
                           help=tooltips.LOSS["predicted"])

    fig = plot_loss_function_curve("binary_cross_entropy", float(y_true),
                                  pred_range=(0.01, 0.99), current_pred=y_pred)
    st.plotly_chart(fig, use_container_width=True)

    eps = 1e-8
    loss_val = -(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))
    st.write(f"**Loss:** {loss_val:.4f}")
    if y_true == 1:
        st.write(f"True label is 1. Predicting {y_pred:.2f} -> loss is {'low' if y_pred > 0.8 else 'high'}.")
    else:
        st.write(f"True label is 0. Predicting {y_pred:.2f} -> loss is {'low' if y_pred < 0.2 else 'high'}.")

# Side-by-side comparison
st.divider()
st.subheader("Key Differences")
col_mse, col_bce = st.columns(2)
with col_mse:
    st.markdown("#### MSE")
    st.write(
        "- Symmetric parabola shape\n"
        "- Gradient is proportional to error\n"
        "- Best for regression (predicting numbers)\n"
        "- Treats over-prediction and under-prediction equally"
    )
with col_bce:
    st.markdown("#### Binary Cross-Entropy")
    st.write(
        "- Asymmetric — steep near 0 and 1\n"
        "- Heavily punishes confident wrong predictions\n"
        "- Best for classification (predicting probabilities)\n"
        "- Works with sigmoid output"
    )

with st.expander("Learn more: Why do different problems need different losses?"):
    st.markdown(explanations.LOSS_INTUITION)

next_step_button("pages/07_backpropagation.py", "Next: Backpropagation")
