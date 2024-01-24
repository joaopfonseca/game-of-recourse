import pandas as pd


def fairness_metrics_viz_data(environment):
    # Get groups
    groups = pd.concat(
        [
            environment.metadata_[step]["X"]["groups"]
            for step in environment.metadata_.keys()
        ]
    )
    groups = groups[~groups.index.duplicated(keep='last')].sort_index()

    # Get time for recourse
    agents_info = environment.analysis.agents_info()
    agents_info = pd.concat([agents_info, groups], axis=1)
    agents_info["time_for_recourse"] = (
        agents_info["favorable_step"] - agents_info["entered_step"]
    )

    agents_info["total_effort"] = (
        agents_info["final_score"] - agents_info["original_score"]
    )

    results = {}

    ai_etr = agents_info  # .dropna(subset="favorable_step")
    if groups.unique().shape[0] != 1:
        # ETR - Effort to recourse
        # ai_etr = ai_etr[ai_etr["n_adaptations"] != 0]

        etr = ai_etr.groupby("groups").mean()["total_effort"]
        results["effort_to_recourse_ratio"] = etr.loc[0] / etr.loc[1]

        # TTR
        ttr = ai_etr.groupby("groups").mean()["time_for_recourse"]
        results["time_to_recourse_difference"] = ttr.loc[0] - ttr.loc[1]

    return ai_etr, results
