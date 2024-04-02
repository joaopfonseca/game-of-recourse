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
pc = st.sidebar.expander("Population", expanded=True)

N_AGENTS = pc.number_input("Initial agents", value=20, help="The number of agents that are created at tilmestep t=0, i.e. the number of agents that exist in the simulation when it begins.")
NEW_AGENTS = pc.number_input("New agents per timestep", value=2, help="The number of new agents that enter at each time step. This number is fixed throughout the entire simulation.")
distribution_type = pc.selectbox(
    "Distribution type",
    options=["Single group", "Biased, no parity", "Biased, some parity"],
    help="We allow data to be generated with three different distributions. Single group is when there is only one group. No parity is when the data is strongly biased, and there is no parity at all between individuals in the groups; imagine Figure 2 in the SIGMOD demo paper, but if all those individuals to the right of the vertical dashed line were not present. Some parity is when data is generated according to the distribution shown in Figure 2."
)
BIAS_FACTOR = pc.number_input("Qualification (bias factor)", value=2, help="The amount of bias that exists between the advantaged and disadvantaged group. It represents the number of standard deviations between the group means.")

#####################################################################
# Environment configurations
#####################################################################
ec = st.sidebar.expander("Environment", expanded=True)

N_LOANS = ec.number_input("Favorable outcomes", value=2, help="The number of individuals that receive a positive outcome at each time step (including time-step 0).")
ADAPTATION = ec.number_input(
    "Global difficulty", value=0.5, min_value=0.0, max_value=1.0, step=0.1, help="Global difficulty is the difficulty of acting on a recourse recommendation, for each agent. For example, it may be easier to act on a recommendation when it is related to appealing a social media ban versus improving one's credit score. Higher value indicate easier recourse."
)

ADAPTATION_TYPE = ec.selectbox(
    "Adaptation type",
    options=["Continuous", "Binary"],
    help="This consideration refers to how faithfully an agent follows the recourse recommendation. Agents may follow the recourse recommendation exactly, or they may outperform (or underperform) the recommendation. In the loan example, if an individual is told to increase their credit score by 50 points, they may do so exactly, or they may actually increase their score by 40 point, or by 60 points. We call the behavior where agents exactly match recourse recommendations “binary” adaptation, and otherwise, we call it “continuous” adaptation."
)

EFFORT_TYPE = ec.selectbox(
    "Effort type",
    options=["Constant", "Flexible"],
    help="This consideration refers to the likelihood of an agent to take any action. An individual that receives a recourse recommendation may or may not act on it. It is determined by several factors, like their implicit willingness to take on challenges or the amount of effort the action requires. For example, if an agent is told to increase their credit score by 20 points to qualify for a loan, they may be more likely to make the effort as opposed to being told to increase it by 200 points. When agents have a fixed willingness to act on recourse we call it “constant” effort, and when agents base their decision to take action on the magnitude of required change, we call it “flexible” effort."
)

BEHAVIOR = (ADAPTATION_TYPE + "_" + EFFORT_TYPE).lower()

TIME_STEPS = ec.number_input("Timesteps", value=10, help="The number of steps of the simulation. Keep in mind the simulation starts at t=0.")

environment_type = ec.selectbox(
    "Bias mitigation strategy",
    options=FRAMEWORKS.keys(),
    help="There are three known strategies for mitigating unfairness in recourse. The first method, proposed by us in recent work, is Circumstance-Normalized Selection (CNA), and is a post-processing intervention based on rank-aware proportional representation. It involves assigning positive outcomes to the highest-scoring individuals from each sub-population, proportionally by population size. The second method is a pre-processing intervention known as Counterfactual Data Augmentation (CDA), and it works by augmenting the initial data with counterfactuals for individuals who received the negative outcome, and then re-training the classifier (or ranker) on this new data. The third mitigation strategy is Group Regularization, which involves re-positioning the decision boundary of a classifier during training to be equidistant from negatively-classified individuals from different groups. As a result, differences in initial circumstances are mitigated. We also allow for a fourth method, which is a combination of CNS and CDA, proposed by us in recent work."
)

ec.markdown("Configure ranker", help="Define the two weighting coefficients for the features. The formula for determining an agent’s score is B_0 * feature 0 + B_1 + feature 1.")

col1, col2 = ec.columns(2)

with col1:
    b1 = st.number_input(r"$\beta_0$", value=0.5, step=0.1)
with col2:
    b2 = st.number_input(r"$\beta_1$", value=0.5, step=0.1)

# "Regularize" coefficients
total = np.abs(b1) + np.abs(b2)
b1 = b1 / total
b2 = b2 / total



random_seed = st.sidebar.number_input("Random state", value=42, help="Control the randomization of the framework.")

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
# The {TOOLKITNAME}
"""
intro = st.expander("See more information")
intro.write(
    f"""
    The {TOOLKITNAME} is an agent-based simulation for synthesizing and analyzing
    real-world algorithmic data. It was built to help practitioners and system-level
    designers improve the reliability and fairness of recourse over time.

    To use the Game of Recourse, simply configure the Population and Environment
    settings for the simulation in the panel on the left side of the screen, and then
    continue through the widgets below.

    To see more information, such as an explanation of each parameter, visit the
    [GitHub repository](https://github.com/joaopfonseca/game-of-recourse).

    This application is based on the paper ["Setting the Right Expectations: Algorithmic
    Recourse over Time"](https://dl.acm.org/doi/pdf/10.1145/3617694.3623251).
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
# Visualize initial population
"""
idd = st.expander("See more information")
idd.markdown(
    """
    This widget provides an overview of the data defined in the "population"
    configuration panel. The initial population data can be visualized in a scatter
    plot (see "Scatter plot" tab), or alternatively, manually inspected (see "Raw data"
    tab).
    """
)


df.groups = df.groups.astype(int).astype(str)
fig = px.scatter(
    df.rename(columns={"f0": "Feature 0", "f1": "Feature 1"}),
    x="Feature 0",
    y="Feature 1",
    color="groups",
)
df.groups = df.groups.astype(float)

tab1, tab2 = st.tabs(["Scatter plot", "Raw data"])
with tab1:
    st.plotly_chart(fig, use_container_width=True)
with tab2:
    df


#####################################################################
# Set up model
#####################################################################
model = IgnoreGroupRanker(np.array([[b1, b2]]), ignore_feature="groups")
y = model.predict(df)

#####################################################################
# Set up environment
#####################################################################
"""
# Explore simulation
"""
intro = st.expander("See more information")
intro.write(
    r"""
    In this widget, the simulation can be visualized in two different interactive
    scatter plots: either as a function of ranker score and timestep, or within the
    feature space (i.e. see how agent features are changing). Use the tabs to move
    between plots. Both visualizations contain a play and stop button to see the
    progression of the agents through the simulation.
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
    df_f.rename(columns={"f0": "Feature 0", "f1": "Feature 1"}),
    x="Feature 0",
    y="Feature 1",
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
# Analyze simulation
"""

intro = st.expander("See more information")
intro.write(
    """
    This widget is split into two tabs. The "Agent info" tab contains a table with
    metadata about every agent that entered the simulation at some point in time. The
    "Environment metrics" tab presents two box plots about the number of timesteps
    required for agents to achieve the outcome (Time For Recourse), the score variation
    incurred to achieve the outcome (Total Effort), and a group-wise Recourse
    Reliability metric. Note that Effort to Recourse ratio and Time to Recourse
    difference are reported, with the disadvantaged population as the reference group.
    This means that effort to Recourse values over 1.0 indicate that the disadvantaged
    group is exerting more effort per successful recourse event than the advantaged
    group. Time to Recourse difference is the literal amount of additional timesteps it
    takes for a member of the disadvantaged population to achieve recourse.
    """
)


metrics, results = fairness_metrics_viz_data(environment)
metrics.index.rename("Agent ID", inplace=True)
metrics.columns = metrics.columns.str.replace("_", " ").str.title()
metrics.Groups = metrics.Groups.astype(int)
metrics.rename(columns={"Time For Recourse": "Time To Recourse"}, inplace=True)

# tab1, tab2, tab3 = st.tabs(
#     ["Agents Info", "Environment Metrics", "Fairness Metrics"]
# )
tab1, tab2 = st.tabs(
    ["Agent info", "Simulation metrics"]
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

    # Recourse Reliability
    df_rr = environment.analysis.success_rate(filter_feature="groups")
    df_rr.columns = df_rr.columns.astype(int).astype(str)
    df_rr = df_rr.fillna(1).mean()

    """
    ________________________
    **Recourse Reliability**
    """
    cols = st.columns(df_rr.shape[0])
    for col_id, i in enumerate(df_rr.index):
        with cols[col_id]:
            st.metric(f"Group {i}", np.round(df_rr.loc[i], 2))

    # df_rr.index = df_rr.index.rename("Group").astype(str)
    # df_rr.rename("Recourse Reliability", inplace=True)

    # fig_rr = px.bar(df_rr.reset_index(), x="Group", y="Recourse Reliability")
    # fig_rr.update_xaxes(dtick=1)
    # st.plotly_chart(fig_rr)

    """
    ____________________
    **Fairness Metrics**
    """
# with tab3:
    if df_f.groups.unique().shape[0] == 1:
        st.warning(
            "Fairness metrics cannot be computed when there is only a single group. "
            "Change to a \"Biased\" distribution type to use this functionality.",
            icon="⚠️"
        )
    else:
        cols = st.columns(len(results))
        for col_id, (key, value) in enumerate(results.items()):
            with cols[col_id]:
                st.metric(key.replace("_", " ").title(), np.round(value, 2))

        # x = [key.replace("_", " ").title() for key in results.keys()]
        # fig_results = go.Figure([go.Bar(x=x, y=list(results.values()))])
        # st.plotly_chart(fig_results)
