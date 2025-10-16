import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator


def plot_metric_at_k(
    metric: Dict[str, Dict],
    parent_save_path: str,
) -> None:
    sns.set_style("darkgrid")

    epochs = len(metric["train"]["loss"])
    names = list(metric["train"].keys())
    for name in names:
        df = pd.DataFrame(
            {
                "value": metric["train"][name] + metric["val"][name],
                "data": ["train"] * epochs + ["val"] * epochs,
                "epochs": [i for i in range(epochs)] * 2,
            }
        )
        plot_metric(
            df=df,
            metric_name=name,
            save_path=os.path.join(parent_save_path, f"{name}.png"),
            hue="data",
        )


def plot_metric(
    df: pd.DataFrame,
    metric_name: str,
    save_path: str,
    hue: Optional[str],
) -> None:
    if hue is not None:
        sns.lineplot(x="epochs", y="value", data=df, hue=hue, marker="o")
        title = f"{metric_name} at every epoch"
    else:
        sns.lineplot(x="epochs", y="value", data=df, marker="o")
        title = f"{metric_name} at every epoch"
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(5))
    plt.ylabel(metric_name)
    plt.title(title)
    plt.savefig(save_path)
    plt.show()
    plt.close()
