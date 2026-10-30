"""
MetaRecom — Flask Web Application
Replaces the previous Streamlit app. Wraps core.pipeline.MetaRecomPipeline
with a clean HTML/CSS/JS frontend (3 themes, sample datasets, history, charts).

Includes optional dataset preprocessing:
- Missing value normalization
- Invalid numeric value handling
- Infinite value handling
- Duplicate removal
- Empty column removal
- Constant column removal
- Missing value imputation

Run:
    pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:5000
"""

import io
import os
import sys
import uuid
import json
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    jsonify,
    abort,
    send_file
)

sys.path.insert(
    0,
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

from core.pipeline import MetaRecomPipeline
from core.meta_learner import ExperimentRepository
from core.data_cleaner import DataCleaner

# Sample dataset loaders
from sklearn.datasets import (
    load_iris,
    load_wine,
    load_breast_cancer,
    load_diabetes
)


# ─── App setup ──────────────────────────────────────────────────────────────

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = (
    25 * 1024 * 1024
)


# In-memory store of recent run results.
# id -> result dictionary
RESULTS_STORE: dict[str, dict] = {}

MAX_STORED = 30


# ─── Helpers ────────────────────────────────────────────────────────────────

def _sanitize_for_json(obj):
    """
    Recursively convert numpy/pandas types so that
    jsonify() and Jinja's tojson can safely serialize them.
    """

    if isinstance(obj, dict):

        return {
            str(k): _sanitize_for_json(v)
            for k, v in obj.items()
        }

    if isinstance(obj, (list, tuple)):

        return [
            _sanitize_for_json(v)
            for v in obj
        ]

    if isinstance(obj, np.integer):

        return int(obj)

    if isinstance(obj, np.floating):

        value = float(obj)

        if np.isnan(value) or np.isinf(value):

            return None

        return value

    if isinstance(obj, np.ndarray):

        return _sanitize_for_json(
            obj.tolist()
        )

    if isinstance(obj, pd.DataFrame):

        return _sanitize_for_json(
            obj.to_dict(
                orient="records"
            )
        )

    if isinstance(obj, pd.Series):

        return _sanitize_for_json(
            obj.to_dict()
        )

    if isinstance(obj, datetime):

        return obj.isoformat()

    return obj


def _load_sample(name: str):
    """
    Return:

        X,
        y,
        dataset_name

    for one of the bundled sklearn datasets.
    """

    name = (
        name or ""
    ).lower()

    loaders = {

        "iris": (
            load_iris,
            "Iris (classification)"
        ),

        "wine": (
            load_wine,
            "Wine (classification)"
        ),

        "breast_cancer": (
            load_breast_cancer,
            "Breast Cancer (classification)"
        ),

        "diabetes": (
            load_diabetes,
            "Diabetes (regression)"
        ),

    }

    if name not in loaders:

        return None

    loader, pretty = loaders[name]

    bunch = loader(
        as_frame=True
    )

    X = bunch.data

    y = bunch.target

    return (
        X,
        y,
        pretty
    )


def _store_result(result: dict) -> str:
    """
    Save a run result.

    If the maximum number of stored runs is exceeded,
    remove the oldest result.
    """

    run_id = uuid.uuid4().hex[:12]

    RESULTS_STORE[run_id] = result

    if len(RESULTS_STORE) > MAX_STORED:

        oldest = next(
            iter(RESULTS_STORE)
        )

        RESULTS_STORE.pop(
            oldest,
            None
        )

    return run_id


def _run_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    dataset_name: str,
    top_k: int,
    cv_folds: int,
    tune: bool,
    compute_mode: str = "auto",
    tune_iter: int = 10
) -> dict:
    """
    Execute the MetaRecom pipeline and return
    a JSON-safe result dictionary.
    """

    pipeline = MetaRecomPipeline(

        top_meta_candidates=top_k,

        cv_folds=cv_folds,

        tune_hyperparams=tune,

        tune_iter=tune_iter,

        compute_mode=compute_mode

    )

    raw = pipeline.run(

        X,

        y,

        dataset_name=dataset_name

    )

    raw["dataset_name"] = dataset_name

    raw["created_at"] = (
        datetime.now().isoformat()
    )

    return _sanitize_for_json(raw)


def _prepare_uploaded_dataset(
    df: pd.DataFrame,
    target_column: str,
    apply_preprocessing: bool
):
    """
    Analyze the uploaded dataset and optionally apply
    automatic preprocessing.

    Returns:

        X
        y
        preprocessing_result
    """

    cleaner = DataCleaner(df)

    # --------------------------------------------------------------
    # ALWAYS ANALYZE THE DATASET
    # --------------------------------------------------------------

    quality_report = (
        cleaner.analyze_dataset()
    )

    preprocessing_result = {

        "applied": apply_preprocessing,

        "quality_report": quality_report,

        "cleaning_summary": None,

        "chart_data": None

    }

    # --------------------------------------------------------------
    # APPLY CLEANING ONLY IF USER REQUESTS IT
    # --------------------------------------------------------------

    if apply_preprocessing:

        cleaned_df, cleaning_summary = (
            cleaner.clean_dataset(

                target_column=target_column

            )
        )

        chart_data = (
            cleaner.get_chart_data(
                cleaning_summary
            )
        )

        preprocessing_result[
            "cleaning_summary"
        ] = cleaning_summary

        preprocessing_result[
            "chart_data"
        ] = chart_data

        df = cleaned_df

    # --------------------------------------------------------------
    # SEPARATE FEATURES AND TARGET
    # --------------------------------------------------------------

    if target_column not in df.columns:

        raise ValueError(
            f"Target column '{target_column}' "
            "was removed during preprocessing."
        )

    y = df[target_column]

    X = df.drop(
        columns=[target_column]
    )

    # Safety check.
    if X.empty:

        raise ValueError(
            "No feature columns remain after preprocessing."
        )

    if y.empty:

        raise ValueError(
            "The target column contains no data."
        )

    return (
        X,
        y,
        preprocessing_result
    )


# ─── Routes: pages ──────────────────────────────────────────────────────────

@app.route("/")
def landing():

    return render_template(

        "landing.html",

        active="home"

    )


@app.route("/app")
def app_page():

    samples = [

        {
            "key": "iris",
            "label": "Iris",
            "task": "Classification",
            "rows": 150,
            "features": 4
        },

        {
            "key": "wine",
            "label": "Wine",
            "task": "Classification",
            "rows": 178,
            "features": 13
        },

        {
            "key": "breast_cancer",
            "label": "Breast Cancer",
            "task": "Classification",
            "rows": 569,
            "features": 30
        },

        {
            "key": "diabetes",
            "label": "Diabetes",
            "task": "Regression",
            "rows": 442,
            "features": 10
        }

    ]

    return render_template(

        "app.html",

        active="app",

        samples=samples

    )


@app.route("/results/<run_id>")
def results_page(run_id):

    result = RESULTS_STORE.get(
        run_id
    )

    if not result:

        abort(

            404,

            description=(
                "Run not found or expired."
            )

        )

    return render_template(

        "results.html",

        active="app",

        run_id=run_id,

        result=result,

        result_json=json.dumps(result)

    )


@app.route("/history")
def history_page():

    repo = ExperimentRepository(

        repo_path=(
            "data/experiment_repo.json"
        )

    )

    df = repo.get_dataframe()

    records = []

    if not df.empty:

        if "timestamp" in df.columns:

            df = df.sort_values(

                "timestamp",

                ascending=False

            )

        for _, row in df.head(200).iterrows():

            records.append({

                "timestamp":
                    row.get(
                        "timestamp",
                        ""
                    ),

                "dataset_name":
                    row.get(
                        "dataset_name",
                        ""
                    ),

                "task_type":
                    row.get(
                        "task_type",
                        ""
                    ),

                "model_name":
                    row.get(
                        "model_name",
                        ""
                    ),

                "score":
                    round(

                        float(

                            row.get(
                                "score",
                                0.0
                            )

                        ),

                        4

                    )

            })

    return render_template(

        "history.html",

        active="history",

        records=records,

        total=len(records)

    )


# ─── Routes: actions ────────────────────────────────────────────────────────

@app.route(
    "/run",
    methods=["POST"]
)
def run_upload():
    """
    Handle:

        CSV upload
        OR sample dataset

    Then optionally preprocess the dataset,
    execute the pipeline,
    and redirect to the results page.
    """

    try:

        # ----------------------------------------------------------
        # FORM SETTINGS
        # ----------------------------------------------------------

        target_col = (

            request.form.get(
                "target_column"
            )

            or ""

        ).strip()

        top_k = int(

            request.form.get(
                "top_k",
                5
            )

        )

        cv_folds = int(

            request.form.get(
                "cv_folds",
                5
            )

        )

        tune = (

            request.form.get(
                "tune"
            )

            == "on"

        )

        compute_mode = (request.form.get("compute_mode", "auto") or "auto").strip().lower()
        if compute_mode not in {"auto", "cpu", "gpu"}:
            compute_mode = "auto"

        try:
            tune_iter = int(request.form.get("tune_iter", 10))
        except (TypeError, ValueError):
            tune_iter = 10
        tune_iter = max(1, min(tune_iter, 30))

        # Checkbox from frontend.
        apply_preprocessing = (

            request.form.get(
                "preprocess"
            )

            == "on"

        )

        sample_key = (

            request.form.get(
                "sample"
            )

            or ""

        ).strip()

        preprocessing_result = None


        # ----------------------------------------------------------
        # SAMPLE DATASET
        # ----------------------------------------------------------

        if sample_key:

            loaded = _load_sample(
                sample_key
            )

            if not loaded:

                return _error_redirect(

                    "Unknown sample dataset: "
                    f"{sample_key}"

                )

            X, y, dataset_name = loaded

            # Build a full dataframe so the cleaner can analyze it.
            full_df = X.copy()

            full_df["target"] = y

            cleaner = DataCleaner(
                full_df
            )

            quality_report = (
                cleaner.analyze_dataset()
            )

            preprocessing_result = {

                "applied":
                    apply_preprocessing,

                "quality_report":
                    quality_report,

                "cleaning_summary":
                    None,

                "chart_data":
                    None

            }

            if apply_preprocessing:

                cleaned_df, cleaning_summary = (
                    cleaner.clean_dataset(

                        target_column="target"

                    )
                )

                chart_data = (
                    cleaner.get_chart_data(
                        cleaning_summary
                    )
                )

                X = cleaned_df.drop(
                    columns=["target"]
                )

                y = cleaned_df["target"]

                preprocessing_result[
                    "cleaning_summary"
                ] = cleaning_summary

                preprocessing_result[
                    "chart_data"
                ] = chart_data


        # ----------------------------------------------------------
        # USER-UPLOADED CSV
        # ----------------------------------------------------------

        else:

            f = request.files.get(
                "file"
            )

            if (

                not f

                or f.filename == ""

            ):

                return _error_redirect(

                    "Please upload a CSV file "
                    "or pick a sample dataset."

                )

            try:

                df = pd.read_csv(
                    f.stream
                )

            except Exception as e:

                return _error_redirect(

                    f"Could not parse CSV: {e}"

                )

            if (

                df.empty

                or df.shape[1] < 2

            ):

                return _error_redirect(

                    "CSV needs at least 2 columns "
                    "(features + target)."

                )

            # Default to last column.
            if not target_col:

                target_col = (
                    df.columns[-1]
                )

            if target_col not in df.columns:

                return _error_redirect(

                    f"Target column "
                    f"'{target_col}' "
                    "not found in CSV."

                )

            dataset_name = (
                os.path.splitext(
                    f.filename
                )[0]
            )

            X, y, preprocessing_result = (
                _prepare_uploaded_dataset(

                    df=df,

                    target_column=target_col,

                    apply_preprocessing=apply_preprocessing

                )
            )


        # ----------------------------------------------------------
        # RUN EXISTING PIPELINE
        # ----------------------------------------------------------

        result = _run_pipeline(

            X,

            y,

            dataset_name,

            top_k,

            cv_folds,

            tune,

            compute_mode,

            tune_iter

        )


        # ----------------------------------------------------------
        # ATTACH PREPROCESSING INFORMATION
        # ----------------------------------------------------------

        result["preprocessing"] = (
            _sanitize_for_json(
                preprocessing_result
            )
        )


        # ----------------------------------------------------------
        # STORE RESULT
        # ----------------------------------------------------------

        run_id = _store_result(
            result
        )

        return redirect(

            url_for(

                "results_page",

                run_id=run_id

            )

        )


    except Exception as e:

        traceback.print_exc()

        return _error_redirect(

            f"Pipeline failed: {e}"

        )


# ─── API ROUTE ──────────────────────────────────────────────────────────────

@app.route(
    "/api/recommend",
    methods=["POST"]
)
def api_recommend():
    """
    JSON endpoint.

    Performs the same process as /run but returns
    the result as JSON instead of redirecting.
    """

    try:

        sample_key = (

            request.form.get(
                "sample",
                ""
            )

        ).strip()

        top_k = int(

            request.form.get(
                "top_k",
                5
            )

        )

        cv_folds = int(

            request.form.get(
                "cv_folds",
                5
            )

        )

        tune = (

            request.form.get(
                "tune",
                "true"
            )

            .lower()

            == "true"

        )

        compute_mode = (request.form.get("compute_mode", "auto") or "auto").strip().lower()
        if compute_mode not in {"auto", "cpu", "gpu"}:
            compute_mode = "auto"

        try:
            tune_iter = int(request.form.get("tune_iter", 10))
        except (TypeError, ValueError):
            tune_iter = 10
        tune_iter = max(1, min(tune_iter, 30))

        apply_preprocessing = (

            request.form.get(
                "preprocess",
                "false"
            )

            .lower()

            == "true"

        )

        preprocessing_result = None


        # ----------------------------------------------------------
        # SAMPLE DATASET
        # ----------------------------------------------------------

        if sample_key:

            loaded = _load_sample(
                sample_key
            )

            if not loaded:

                return jsonify({

                    "error":
                        f"Unknown sample: "
                        f"{sample_key}"

                }), 400

            X, y, dataset_name = loaded

            full_df = X.copy()

            full_df["target"] = y

            cleaner = DataCleaner(
                full_df
            )

            quality_report = (
                cleaner.analyze_dataset()
            )

            preprocessing_result = {

                "applied":
                    apply_preprocessing,

                "quality_report":
                    quality_report,

                "cleaning_summary":
                    None,

                "chart_data":
                    None

            }

            if apply_preprocessing:

                cleaned_df, cleaning_summary = (
                    cleaner.clean_dataset(

                        target_column="target"

                    )
                )

                chart_data = (
                    cleaner.get_chart_data(
                        cleaning_summary
                    )
                )

                X = cleaned_df.drop(
                    columns=["target"]
                )

                y = cleaned_df["target"]

                preprocessing_result[
                    "cleaning_summary"
                ] = cleaning_summary

                preprocessing_result[
                    "chart_data"
                ] = chart_data


        # ----------------------------------------------------------
        # USER-UPLOADED DATASET
        # ----------------------------------------------------------

        else:

            f = request.files.get(
                "file"
            )

            if not f:

                return jsonify({

                    "error":
                        "No file uploaded"

                }), 400

            df = pd.read_csv(
                f.stream
            )

            if (

                df.empty

                or df.shape[1] < 2

            ):

                return jsonify({

                    "error":
                        "CSV needs at least "
                        "2 columns."

                }), 400

            target_column = (

                request.form.get(
                    "target_column"
                )

                or df.columns[-1]

            )

            if target_column not in df.columns:

                return jsonify({

                    "error":
                        f"Target column "
                        f"'{target_column}' "
                        "not found."

                }), 400

            dataset_name = (
                os.path.splitext(
                    f.filename
                )[0]
            )

            X, y, preprocessing_result = (
                _prepare_uploaded_dataset(

                    df=df,

                    target_column=target_column,

                    apply_preprocessing=apply_preprocessing

                )
            )


        # ----------------------------------------------------------
        # RUN PIPELINE
        # ----------------------------------------------------------

        result = _run_pipeline(

            X,

            y,

            dataset_name,

            top_k,

            cv_folds,

            tune,

            compute_mode,

            tune_iter

        )

        result["preprocessing"] = (
            _sanitize_for_json(
                preprocessing_result
            )
        )

        run_id = _store_result(
            result
        )

        return jsonify({

            "run_id":
                run_id,

            "result":
                result

        })


    except Exception as e:

        traceback.print_exc()

        return jsonify({

            "error":
                str(e)

        }), 500

@app.route("/api/inspect-csv", methods=["POST"])
def inspect_csv():
    """Inspect an uploaded CSV and return its columns and suggested target."""
    try:
        f = request.files.get("file")

        if f is None or f.filename == "":
            return jsonify({
                "error": "No CSV file was uploaded."
            }), 400

        try:
            df = pd.read_csv(f.stream)
        except Exception as e:
            return jsonify({
                "error": f"Could not read CSV: {str(e)}"
            }), 400

        if df.empty:
            return jsonify({
                "error": "The uploaded CSV is empty."
            }), 400

        columns = [str(column) for column in df.columns]

        # Common names that are likely to be target columns
        target_names = [
            "target",
            "label",
            "class",
            "output",
            "outcome",
            "result",
            "response",
            "diagnosis",
            "heart_disease",
            "disease",
            "survived",
            "price",
            "quality",
            "y"
        ]

        # Create case-insensitive lookup
        normalized_columns = {
            str(column).strip().lower(): str(column)
            for column in df.columns
        }

        suggested_target = None

        # First try common target names
        for name in target_names:
            if name in normalized_columns:
                suggested_target = normalized_columns[name]
                break

        # Otherwise use the last column
        if suggested_target is None and columns:
            suggested_target = columns[-1]

        return jsonify({
            "columns": columns,
            "suggested_target": suggested_target,
            "rows": int(df.shape[0]),
            "features": int(df.shape[1])
        })

    except Exception as e:
        traceback.print_exc()

        return jsonify({
            "error": f"Failed to inspect CSV: {str(e)}"
        }), 500
# ─── Error helpers ──────────────────────────────────────────────────────────

def _error_redirect(msg: str):
    """
    Redirect back to the application page
    with an error message.
    """

    return redirect(

        url_for(
            "app_page"
        )

        + f"?error={msg}"

    )


@app.errorhandler(404)
def not_found(e):

    return render_template(

        "base.html",

        active="",

        error=(
            "404 — Page not found"
        )

    ), 404


@app.errorhandler(413)
def too_large(e):

    return _error_redirect(

        "File too large "
        "(max 25 MB)."

    )


@app.route("/favicon.ico")
def favicon():

    return "", 204
@app.route("/api/dataset-columns", methods=["POST"])
def dataset_columns():
    """
    Inspect an uploaded CSV and return:
    - all column names
    - a recommended target column
    """

    try:

        f = request.files.get("file")

        if not f or f.filename == "":
            return jsonify({
                "error": "No CSV file uploaded."
            }), 400


        # Read only the first 200 rows.
        # We only need enough data to inspect the dataset.
        df = pd.read_csv(
            f,
            nrows=200
        )


        if df.empty:
            return jsonify({
                "error": "The uploaded CSV is empty."
            }), 400


        columns = df.columns.tolist()


        if not columns:
            return jsonify({
                "error": "No columns found in the CSV."
            }), 400


        # ============================================================
        # TARGET COLUMN RECOMMENDATION
        # ============================================================

        priority_names = [

            "target",

            "label",

            "class",

            "output",

            "outcome",

            "result",

            "response",

            "prediction",

            "heart_disease",

            "disease",

            "diagnosis",

            "survived",

            "churn"

        ]


        recommended = None


        # ------------------------------------------------------------
        # Exact match
        # ------------------------------------------------------------

        for priority in priority_names:

            for column in columns:

                if (
                    column
                    .strip()
                    .lower()
                    == priority
                ):

                    recommended = column

                    break


            if recommended:

                break


        # ------------------------------------------------------------
        # Partial match
        # ------------------------------------------------------------

        if not recommended:

            for priority in priority_names:

                for column in columns:

                    if priority in column.strip().lower():

                        recommended = column

                        break


                if recommended:

                    break


        # ------------------------------------------------------------
        # Fallback
        # ------------------------------------------------------------

        if not recommended:

            recommended = columns[-1]


        # ============================================================
        # RETURN JSON
        # ============================================================

        return jsonify({

            "columns": columns,

            "recommended_target": recommended

        })


    except Exception as e:

        traceback.print_exc()

        return jsonify({

            "error": str(e)

        }), 500
    
# ─── Run application ────────────────────────────────────────────────────────

if __name__ == "__main__":

    app.run(

        host="127.0.0.1",

        port=5000,

        debug=True,

        use_reloader=False

    )