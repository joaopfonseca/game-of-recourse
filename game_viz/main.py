import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from mlresearch.utils import set_matplotlib_style
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
from metrics import fairness_metrics_viz_data

set_matplotlib_style()

TOOLKITNAME = "Game of Recourse"

st.set_page_config(
        page_title=TOOLKITNAME,
)

FRAMEWORKS = {
    "No bias mitigation": BaseEnvironment,
    "Circumstance-Normalized Selection": FairEnvironment,
    "Counterfactual Data Augmentation": CDAEnvironment,
    "Group Regularization": ModelRetrainEnvironment,
    "Combined (CNS + CDA)": FairCDAEnvironment,
}


#####################################################################
# Global configurations
#####################################################################
st.sidebar.markdown("# Configurations")

#####################################################################
# Population configurations
#####################################################################
pc = st.sidebar.expander("Population", expanded=False)

N_AGENTS = pc.number_input("Initial agents", value=20)
NEW_AGENTS = pc.number_input("New agents per timestep", value=2)
distribution_type = pc.selectbox(
    "Distribution type",
    options=["Single group", "Biased, no parity", "Biased, some parity"],
)
BIAS_FACTOR = pc.number_input("Qualification (bias factor)", value=2)

#####################################################################
# Environment configurations
#####################################################################
ec = st.sidebar.expander("Environment", expanded=False)

N_LOANS = ec.number_input("Favorable outcomes", value=2)
ADAPTATION = ec.number_input(
    "Global difficulty", value=0.5, min_value=0.0, max_value=1.0, step=0.1
)

ADAPTATION_TYPE = ec.selectbox(
    "Adaptation type",
    options=["Continuous", "Binary"],
)

EFFORT_TYPE = ec.selectbox(
    "Effort type",
    options=["Constant", "Flexible"],
)

BEHAVIOR = (ADAPTATION_TYPE + "_" + EFFORT_TYPE).lower()

TIME_STEPS = ec.number_input("Timesteps", value=10)

environment_type = ec.selectbox(
    "Bias mitigation strategy",
    options=FRAMEWORKS.keys(),
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

f"""
# Welcome to the {TOOLKITNAME}!
"""
intro = st.expander("See more information")
intro.write(
    f"""
    "{TOOLKITNAME}" is a simulator of recourse-providing environments, where agents
    compete to obtain a scarce outcome, determined by an automated system (Machine
    Learning classifier or otherwise). At each timestep, agents that fail to receive it
    will receive feedback (*i.e.*, algorithmic recourse) on why they failed, and
    what they should do to improve.


    TODO (regarding text):
    - Add authors list
    - Proper intro
    - Explanations for each component in the main body
    - Add formula in Ranker (use latex)
    """
)


#####################################################################
# Generate data
#####################################################################
rng = np.random.default_rng(random_seed)

if distribution_type == "Single group":
    scaler_func = get_scaler
    data_gen_func = generate_synthetic_data

elif distribution_type == "Biased, no parity":
    scaler_func = get_scaler
    data_gen_func = biased_data_generator

elif distribution_type == "Biased, some parity":
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

#####################################################################
# Initial data distribution
#####################################################################
"""
# Initial data distribution
"""
idd = st.expander("See more information")
idd.markdown(
    """
    Explanation on the choice of the data distribution goes here.

    The dataframe should be editable as well.
    """
)


df.groups = df.groups.astype(int).astype(str)
fig = px.scatter(
    df,
    x="f0",
    y="f1",
    color="groups",
)
df.groups = df.groups.astype(float)

tab1, tab2 = st.tabs(["Scatter plot", "Data"])
with tab1:
    st.plotly_chart(fig, use_container_width=True)
with tab2:
    df


#####################################################################
# Set up model
#####################################################################
"""
# Set up Ranker
"""
idd = st.expander("See more information")
idd.markdown(
    """
    Explanation on the ranker parameters goes here.

    The dataframe should be editable as well.
    """
)

col1, col2 = st.columns(2)

with col1:
    b1 = st.number_input(r"$\beta_0$", value=0.5, step=0.1)
with col2:
    b2 = st.number_input(r"$\beta_1$", value=0.5, step=0.1)

# "Regularize" coefficients
total = np.abs(b1) + np.abs(b2)
b1 = b1 / total
b2 = b2 / total

model = IgnoreGroupRanker(np.array([[b1, b2]]), ignore_feature="groups")
y = model.predict(df)

#####################################################################
# Set up environment
#####################################################################
"""
# Run environment
"""
intro = st.expander("See more information")
intro.write(
    r"""
    Description goes here.
    """
)


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

if environment_type in ["No bias mitigation", "Circumstance-Normalized Selection"]:
    pass

elif environment_type in ["Counterfactual Data Augmentation", "Combined (CNS + CDA)"]:
    model = IgnoreGroupLR(ignore_feature="groups", random_state=random_seed)
    model.fit(df, y)

elif environment_type == "Group Regularization":
    model = RecourseAwareClassifer(
        LogisticRegression(random_state=random_seed),
        l=100,
        niter=100,
        group_feature="groups",
    )
    model.fit(df, y)

recourse = NFeatureRecourse(
    model=model,
    threshold=0.5,
    categorical=["groups"],
)
recourse.set_actions(df)
recourse.action_set_.lb = 0.0
recourse.action_set_.ub = 1.0

kwargs = (
    {}
    if environment_type
    in ["No bias mitigation", "Group Regularization", "Counterfactual Data Augmentation"]
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
    random_state=random_seed,
    **kwargs,
)

environment.simulate(TIME_STEPS)

#####################################################################
# Visualizations
#####################################################################

# fig, ax = plt.subplots(1, 1)
# environment.plot.agent_scores(ax=ax)
# st.pyplot(fig)

dfs = []
for i in range(0, TIME_STEPS + 1):
    df = environment.metadata_[i]["X"].copy()
    df["Score"] = environment.metadata_[i]["score"]
    df["Timestep"] = i
    df["outcome"] = environment.metadata_[i]["outcome"]
    df = df.reset_index(names="agent_id")
    df.sort_values(by=["Score"], inplace=True, ascending=False)
    df = df.reset_index(drop=True)
    df["color"] = df.apply(
        lambda row: (
            "Group = "
            + str(int(row["groups"]))
            + " | "
            + "Outcome = "
            + str(int(row["outcome"]))
        ),
        axis=1,
    )
    dfs.append(df)

df_f = pd.concat(dfs, ignore_index=True)


def SetColor(x):
    if x == "00":
        return "lightblue"
    elif x == "10":
        return "darkblue"
    elif x == "01":
        return "lightgreen"
    elif x == "11":
        return "darkgreen"


fig_scores = px.scatter(
    df_f,
    x="Timestep",
    y="Score",
    animation_frame="Timestep",
    animation_group="agent_id",
    color="color",
    color_discrete_map={
        "00": "lightblue",
        "10": "darkblue",
        "01": "lightgreen",
        "11": "darkgreen",
    },
    log_x=False,
    size_max=55,
    range_x=[0, TIME_STEPS],
    range_y=[0, 1],
)  # .update_traces(marker=dict(color=list(map(SetColor, df_f['color']))))
fig_scores.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1000


fig_features = px.scatter(
    df_f,
    x="f0",
    y="f1",
    animation_frame="Timestep",
    animation_group="agent_id",
    color="color",
    color_discrete_map={
        "00": "lightblue",
        "10": "darkblue",
        "01": "lightgreen",
        "11": "darkgreen",
    },
    log_x=False,
    size_max=55,
    range_x=[0, 1],
    range_y=[0, 1],
)  # .update_traces(marker=dict(color=list(map(SetColor, df_f['color']))))
fig_features.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1000

tab1, tab2, = st.tabs(
    ["Scores", "Features"]
)
with tab1:
    st.plotly_chart(fig_scores)
with tab2:
    st.plotly_chart(fig_features)

df_download = df_f.drop(columns="color").copy()
df_download.groups = df_download.groups.astype(int)
st.sidebar.download_button(
    "Download data",
    df_download.to_csv(index=False),
    file_name="recourse_game_simulation.csv",
)

#####################################################################
# Metrics
#####################################################################
"""
# Simulation analysis
"""

metrics, results = fairness_metrics_viz_data(environment)
metrics.index.rename("Agent ID", inplace=True)
metrics.columns = metrics.columns.str.replace("_", " ").str.title()
metrics.Groups = metrics.Groups.astype(int)
metrics.rename(columns={"Time For Recourse": "Time To Recourse"}, inplace=True)

tab1, tab2, tab3 = st.tabs(
    ["Agents Info", "Environment Metrics", "Fairness Metrics"]
)
with tab1:
    st.write(
        metrics
    )

with tab2:
    cols = ["Time To Recourse", "Total Effort"]
    fig_metrics = make_subplots(rows=2, cols=1, subplot_titles=cols)

    for i, col in enumerate(cols):
        _fig = px.box(metrics.sort_values("Groups"), x=col, color="Groups")
        for t in _fig.data:
            t.update(showlegend=(not bool(i)))
            fig_metrics.add_trace(t, row=i+1, col=1)

    fig_metrics.update_layout(
        boxmode="group", margin={"l": 0, "r": 0, "t": 20, "b": 0}
    )

    st.plotly_chart(fig_metrics)

    # Success rate
    df_rr = environment.analysis.success_rate(filter_feature="groups")
    df_rr.columns = df_rr.columns.astype(int).astype(str)
    df_rr = df_rr.fillna(1).mean()
    df_rr.index = df_rr.index.rename("Group").astype(str)
    df_rr.rename("Recourse Reliability", inplace=True)

    fig_rr = px.bar(df_rr.reset_index(), x="Group", y="Recourse Reliability")
    fig_rr.update_xaxes(dtick=1)
    st.plotly_chart(fig_rr)

with tab3:
    if df_f.groups.unique().shape[0] == 1:
        st.warning(
            "Fairness metrics cannot be computed when there is only a single group. "
            "Change to a \"Biased\" distribution type to use this functionality.",
            icon="⚠️"
        )
    else:
        # results = pd.Series(results).to_frame().reset_index()
        x = [key.replace("_", " ").title() for key in results.keys()]
        fig_results = go.Figure([go.Bar(x=x, y=list(results.values()))])
        st.plotly_chart(fig_results)
