import streamlit as st

from components.state_manager import init_state

st.set_page_config(
    page_title="Deep Learning Explainer",
    page_icon=":material/neurology:",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_state()

pages = {
    "Getting Started": [
        st.Page("pages/01_welcome.py", title="Welcome", icon=":material/school:"),
    ],
    "The Basics": [
        st.Page("pages/02_the_neuron.py", title="The Neuron", icon=":material/circle:"),
        st.Page("pages/03_activation_functions.py", title="Activation Functions", icon=":material/show_chart:"),
        st.Page("pages/04_build_a_network.py", title="Build a Network", icon=":material/account_tree:"),
        st.Page("pages/05_forward_pass.py", title="Forward Pass", icon=":material/arrow_forward:"),
    ],
    "Learning": [
        st.Page("pages/06_loss_functions.py", title="Loss Functions", icon=":material/trending_down:"),
        st.Page("pages/07_backpropagation.py", title="Backpropagation", icon=":material/replay:"),
        st.Page("pages/08_training_loop.py", title="Training Loop", icon=":material/loop:"),
    ],
    "Advanced": [
        st.Page("pages/09_optimizers.py", title="Optimizers", icon=":material/speed:"),
        st.Page("pages/10_overfitting.py", title="Overfitting & Regularization", icon=":material/tune:"),
    ],
    "Capstone": [
        st.Page("pages/11_digit_recognition.py", title="Digit Recognition", icon=":material/draw:"),
    ],
    "Wrap Up": [
        st.Page("pages/12_whats_next.py", title="What's Next", icon=":material/rocket_launch:"),
    ],
}

page = st.navigation(pages)
page.run()
