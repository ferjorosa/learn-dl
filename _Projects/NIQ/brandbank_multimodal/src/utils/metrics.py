import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
)


def calculate_metrics(df, label_colname, prediction_colname):
    y_true = df[label_colname].astype(str)
    y_pred = df[prediction_colname].astype(str)

    accuracy = accuracy_score(y_true, y_pred)
    precision_score_macro = precision_score(y_true, y_pred, average="macro")
    recall_score_macro = recall_score(y_true, y_pred, average="macro")
    f1_score_macro = f1_score(y_true, y_pred, average="macro")

    metrics_dict = {
        "Accuracy": accuracy,
        "Recall (macro)": recall_score_macro,
        "Precision (macro)": precision_score_macro,
        "F1 (macro)": f1_score_macro,
    }

    metrics_df = pd.DataFrame([metrics_dict])

    return metrics_df


def top_10_confusions(df, label_colname, prediction_colname):
    df = df[df[label_colname] != df[prediction_colname]]
    confusion_counts = (
        df.groupby([label_colname, prediction_colname])
        .size()
        .reset_index(name="count")
    )
    top_10_confusions = (
        confusion_counts.sort_values("count", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    return top_10_confusions
