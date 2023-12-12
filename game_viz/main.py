import streamlit as st
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from mlresearch.utils import set_matplotlib_style

from recgame.environments import (
    BaseEnvironment,
    ModelRetrainEnvironment,
    CDAEnvironment,
    FairEnvironment,
    FairCDAEnvironment,
)

from data_generators import (
    generate_synthetic_data,
    get_scaler_hc,
    biased_data_generator_hc,
    get_scaler,
    biased_data_generator,
)
from scorers import IgnoreGroupRanker, IgnoreGroupLR, RecourseAwareClassifer
from recourse import NFeatureRecourse

set_matplotlib_style()

FRAMEWORKS = {
    "Basic": BaseEnvironment,
    "Fair Selection": FairEnvironment,
    "CDA": CDAEnvironment,
    "Group Regularization": ModelRetrainEnvironment,
    "Combined (Fair Selection + CDA)": FairCDAEnvironment,
}


"""
# Generate some data

This should be a collapsable section. The dataframe should be editable as well.
"""

#####################################################################
# Global configurations
#####################################################################
st.sidebar.markdown("# Configurations")

#####################################################################
# Population configurations
#####################################################################
pc = st.sidebar.expander("Population", expanded=False)

distribution_type = pc.selectbox(
    "Distribution type", options=["Basic", "Biased 1", "Biased 2 (hc)"]
)
N_AGENTS = pc.number_input("Initial Agents", value=60)
NEW_AGENTS = pc.number_input("New agents per time step", value=6)
BIAS_FACTOR = pc.number_input("Bias factor", value=2)

#####################################################################
# Environment configurations
#####################################################################
ec = st.sidebar.expander("Environment", expanded=False)

environment_type = ec.selectbox(
    "Environment type",
    options=FRAMEWORKS.keys(),
)

N_LOANS = ec.number_input("Favorable outcomes", value=6)
ADAPTATION = ec.number_input(
    "Adaptation rate", value=0.5, min_value=0.0, max_value=1.0, step=0.1
)

BEHAVIOR = (
    ec.selectbox(
        "Behavior Function",
        options=["Continuous/Constant", "Binary/Constant", "Continuous/Flexible"],
    )
    .lower()
    .replace("/", "_")
)

random_seed = st.sidebar.number_input("Random state", value=42)

#####################################################################
#####################################################################
#####################################################################
#####################################################################
# Main
#####################################################################
#####################################################################
#####################################################################
#####################################################################


#####################################################################
# Generate data
#####################################################################
rng = np.random.default_rng(random_seed)

if distribution_type == "Basic":
    scaler_func = get_scaler
    data_gen_func = generate_synthetic_data

elif distribution_type == "Biased 1":
    scaler_func = get_scaler
    data_gen_func = biased_data_generator

elif distribution_type == "Biased 2 (hc)":
    scaler_func = get_scaler_hc
    data_gen_func = biased_data_generator_hc


scaler = scaler_func(
    n_agents=10_000,
    bias_factor=BIAS_FACTOR,
    random_state=rng,
    N_AGENTS=N_AGENTS,
    N_LOANS=N_LOANS,
)

df = data_gen_func(
    N_AGENTS,
    n_continuous=2,
    bias_factor=BIAS_FACTOR,
    scaler=scaler,
    random_state=rng,
    N_LOANS=N_LOANS,
    N_AGENTS=N_AGENTS,
)

c = (
    alt.Chart(df)
    .mark_circle()
    .encode(x="f0", y="f1", color="groups", tooltip=["f0", "f1", "groups"])
)

st.altair_chart(c, use_container_width=True)


#####################################################################
# Set up model
#####################################################################
"""
# Set up Ranker
"""
col1, col2 = st.columns(2)

with col1:
    b1 = st.number_input(r"$\beta_0$", value=0.5, min_value=-2.0, max_value=2.0, step=0.1)
with col2:
    b2 = st.number_input(r"$\beta_1$", value=0.5, min_value=-2.0, max_value=2.0, step=0.1)

if b1+b2 != 0:
    b1 = b1 / (b1 + b2)
    b2 = b2 / (b1 + b2)

model = IgnoreGroupRanker(np.array([[b1, b2]]), ignore_feature="groups")
y = model.predict(df)

# st.write(model)


#####################################################################
# Set up environment
#####################################################################
"""
# Run environment
"""


def env_data_generator(n_agents):
    return data_gen_func(
        n_agents,
        n_continuous=2,
        bias_factor=BIAS_FACTOR,
        scaler=scaler,
        random_state=rng,
        N_LOANS=N_LOANS,
        N_AGENTS=N_AGENTS,
    )


environment = FRAMEWORKS[environment_type]

if environment_type in ["Basic", "Fair Selection"]:
    pass

elif environment_type in ["CDA", "Combined (Fair Selection + CDA)"]:
    model = IgnoreGroupLR(ignore_feature="groups", random_state=random_seed)
    model.fit(df, y)

elif environment_type == "Group Regularization":
    model = RecourseAwareClassifer(
        LogisticRegression(random_state=random_seed),
        l=100,
        niter=100,
        group_feature="groups"
    )
    model.fit(df, y)

recourse = NFeatureRecourse(
    model=model,
    threshold=0.5,
    categorical=["groups"],
)

kwargs = (
    {}
    if environment_type in ["Basic", "Group Regularization", "CDA"]
    else {"group_feature": "groups"}
)

environment = environment(
    X=df,
    recourse=recourse,
    data_source_func=env_data_generator,
    threshold=N_LOANS,
    threshold_type="absolute",
    growth_rate=NEW_AGENTS,
    growth_rate_type="absolute",
    adaptation=ADAPTATION,
    behavior_function=BEHAVIOR,
    **kwargs
)

environment.simulate(20)

fig, ax = plt.subplots(1, 1)
environment.plot.agent_scores(ax=ax)
st.pyplot(fig)
