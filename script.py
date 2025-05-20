import os
from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import timedelta

import implicit
import mlflow
import numpy as np
import polars as pl
from dotenv import load_dotenv
from scipy.sparse import csr_matrix

load_dotenv()

K = 40
EVAL_DAYS_TRESHOLD = 14
DATA_DIR = "data/"


@dataclass
class AlsConfig:
    als_ground_truth_value1: float
    als_ground_truth_value2: float
    als_time_delta: float
    als_time_weight: bool
    als_factors: int
    als_regularization: float
    als_alpha: float
    als_iterations: int
    random_state: int


def get_data():
    df_test_users = pl.read_parquet(f"{DATA_DIR}/test_users.pq")
    df_clickstream = pl.read_parquet(f"{DATA_DIR}/clickstream.pq")
    df_event = pl.read_parquet(f"{DATA_DIR}/events.pq")
    return df_test_users, df_clickstream, df_event


def split_train_test(df_clickstream: pl.DataFrame, df_event: pl.DataFrame):
    treshhold = df_clickstream["event_date"].max() - timedelta(days=EVAL_DAYS_TRESHOLD)

    # Train data
    df_train = df_clickstream.filter(df_clickstream["event_date"] <= treshhold)
    df_train = df_train.with_columns(
        (treshhold - pl.col("event_date"))
        .dt.total_hours()
        .alias("event_interval_hours")
    )
    df_train = df_train.with_columns(
        pl.col("event")
        .is_in(df_event.filter(pl.col("is_contact") == 1)["event"].unique())
        .alias("is_contact")
    )
    # Eval data
    df_eval = df_clickstream.filter(df_clickstream["event_date"] > treshhold)[
        ["cookie", "node", "event"]
    ]
    df_eval = df_eval.join(df_train, on=["cookie", "node"], how="anti")
    df_eval = df_eval.filter(
        pl.col("event").is_in(
            df_event.filter(pl.col("is_contact") == 1)["event"].unique()
        )
    )
    df_eval = df_eval.filter(
        pl.col("cookie").is_in(df_train["cookie"].unique())
    ).filter(pl.col("node").is_in(df_train["node"].unique()))
    df_eval = df_eval.unique(["cookie", "node"])

    return df_train, df_eval


def get_als_pred(
    users: pl.Series,
    nodes: pl.Series,
    values: pl.Series,
    is_contact: pl.Series,
    user_to_pred: pl.Series,
    als_config: AlsConfig,
):
    user_ids = users.unique().to_list()
    item_ids = nodes.unique().to_list()
    values = values.to_list()

    user_id_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_id_to_index = {item_id: idx for idx, item_id in enumerate(item_ids)}
    index_to_item_id = {v: k for k, v in item_id_to_index.items()}

    rows = users.replace_strict(user_id_to_index).to_list()
    cols = nodes.replace_strict(item_id_to_index).to_list()

    if als_config.als_time_weight:
        values = [
            (
                als_config.als_ground_truth_value1
                * np.exp(-als_config.als_time_delta * val)
                if is_contact[i]
                else als_config.als_ground_truth_value2
                * np.exp(-als_config.als_time_delta * val)
            )
            for i, val in enumerate(values)
        ]
    else:
        values = [
            (
                als_config.als_ground_truth_value1
                if contact
                else als_config.als_ground_truth_value2
            )
            for contact in is_contact
        ]

    sparse_matrix = csr_matrix(
        (values, (rows, cols)), shape=(len(user_ids), len(item_ids))
    )

    model = implicit.als.AlternatingLeastSquares(
        factors=als_config.als_factors,
        regularization=als_config.als_regularization,
        alpha=als_config.als_alpha,
        iterations=als_config.als_iterations,
        random_state=42,
    )
    model.fit(
        sparse_matrix,
    )

    user4pred = np.array([user_id_to_index[i] for i in user_to_pred])

    recommendations, scores = model.recommend(
        user4pred, sparse_matrix[user4pred], N=K, filter_already_liked_items=True
    )

    df_pred = pl.DataFrame(
        {
            "node": [
                [index_to_item_id[i] for i in i] for i in recommendations.tolist()
            ],
            "cookie": list(user_to_pred),
            "scores": scores.tolist(),
        }
    )
    df_pred = df_pred.explode(["node", "scores"])
    return df_pred, model


def train(df_train: pl.DataFrame, df_eval: pl.DataFrame, als_config: AlsConfig):
    users = df_train["cookie"]
    nodes = df_train["node"]
    eval_users = df_eval["cookie"].unique().to_list()
    values = df_train["event_interval_hours"]
    is_contact = df_train["is_contact"]
    df_pred, _ = get_als_pred(users, nodes, values, is_contact, eval_users, als_config)
    return df_pred


def recall_at(df_true, df_pred, k=K):
    return (
        df_true[["node", "cookie"]]
        .join(
            df_pred.group_by("cookie")
            .head(k)
            .with_columns(value=1)[["node", "cookie", "value"]],
            how="left",
            on=["cookie", "node"],
        )
        .select([pl.col("value").fill_null(0), "cookie"])
        .group_by("cookie")
        .agg([pl.col("value").sum() / pl.col("value").count()])["value"]
        .mean()
    )


def main():
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))

    parser = ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="homework-ivmerkulov")
    parser.add_argument("--run_name", type=str, default="als_contact_airflow")
    parser.add_argument("--als_ground_truth_value1", type=float, default=1.0)
    parser.add_argument("--als_ground_truth_value2", type=float, default=1.0)
    parser.add_argument("--als_time_delta", type=float, default=1e-9)
    parser.add_argument("--als_time_weight", type=bool, default=True)
    parser.add_argument("--als_factors", type=int, default=60)
    parser.add_argument("--als_regularization", type=float, default=0.01)
    parser.add_argument("--als_alpha", type=float, default=0.01)
    parser.add_argument("--als_iterations", type=int, default=1)
    parser.add_argument("--random_state", type=int, default=42)

    args = parser.parse_args()

    als_config = AlsConfig(
        als_ground_truth_value1=args.als_ground_truth_value1,
        als_ground_truth_value2=args.als_ground_truth_value1,
        als_time_delta=args.als_time_delta,
        als_time_weight=args.als_time_weight,
        als_factors=args.als_factors,
        als_regularization=args.als_regularization,
        als_alpha=args.als_alpha,
        als_iterations=args.als_iterations,
        random_state=args.random_state,
    )

    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=args.run_name):
        _, df_clickstream, df_event = get_data()
        df_train, df_eval = split_train_test(df_clickstream, df_event)
        df_pred = train(df_train, df_eval, als_config)

        recall_40 = recall_at(df_eval, df_pred, k=K)

        mlflow.log_params(
            params={
                "als_factors": args.als_factors,
                "als_iterations": args.als_iterations,
                "als_regularization": args.als_regularization,
                "als_alpha": args.als_alpha,
                "als_ground_truth_value": args.als_ground_truth_value1,
                "als_ground_truth_value_contact": args.als_ground_truth_value2,
                "als_data_prepare": args.als_data_prepare,
                "random_state": args.random_state,
                "model": "als",
            }
        )
        if args.als_time_weight:
            mlflow.log_param("als_time_delta", args.als_time_delta)
        mlflow.log_metrics(metrics={"Recall_40": recall_40})

        # with open("als_model.pkl", "wb") as f:
        #    pickle.dump(model, f)
        #
        # mlflow.log_artifact("als_model.pkl")


if __name__ == "__main__":
    main()
