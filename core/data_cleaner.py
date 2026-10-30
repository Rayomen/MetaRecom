import pandas as pd
import numpy as np


class DataCleaner:
    """
    Handles dataset quality analysis and optional automatic preprocessing.

    The class first analyzes the dataset without modifying it.
    Cleaning operations are only applied when explicitly requested.
    """

    MISSING_INDICATORS = {
        "",
        " ",
        "  ",
        "na",
        "n/a",
        "nan",
        "none",
        "null",
        "missing",
        "?",
        "-"
    }

    def __init__(self, df):
        """
        Initialize the cleaner with a copy of the dataset.

        Parameters
        ----------
        df : pandas.DataFrame
            The uploaded dataset.
        """

        self.original_df = df.copy()
        self.df = df.copy()

        self.analysis = {}
        self.cleaning_log = []

    def _normalize_missing_values(self, df):
        """
        Convert common textual representations of missing values
        into actual NumPy NaN values.

        This method does not modify the original dataset.
        """

        cleaned_df = df.copy()

        for column in cleaned_df.columns:

            if cleaned_df[column].dtype == "object":

                cleaned_df[column] = (
                    cleaned_df[column]
                    .astype(str)
                    .str.strip()
                )

                cleaned_df[column] = cleaned_df[column].replace(
                    list(self.MISSING_INDICATORS),
                    np.nan
                )

        return cleaned_df
    def analyze_dataset(self):
        """
        Analyze the dataset and identify potential data quality issues.

        This method does not modify self.df or the original dataset.
        It only returns a report describing detected issues.
        """

        # Work on a temporary normalized copy so that textual
        # missing indicators can be detected without changing the dataset.
        analysis_df = self._normalize_missing_values(self.original_df)

        total_rows = len(analysis_df)
        total_columns = len(analysis_df.columns)

        # ----------------------------------------------------------
        # MISSING VALUES
        # ----------------------------------------------------------

        missing_by_column = (
            analysis_df.isna()
            .sum()
        )

        total_missing = int(missing_by_column.sum())

        missing_columns = {
            column: int(count)
            for column, count in missing_by_column.items()
            if count > 0
        }

        # ----------------------------------------------------------
        # DUPLICATE ROWS
        # ----------------------------------------------------------

        duplicate_rows = int(
            analysis_df.duplicated().sum()
        )

        # ----------------------------------------------------------
        # INFINITE VALUES
        # ----------------------------------------------------------

        numeric_columns = analysis_df.select_dtypes(
            include=[np.number]
        ).columns

        infinite_by_column = {}

        for column in numeric_columns:

            infinite_count = int(
                np.isinf(analysis_df[column]).sum()
            )

            if infinite_count > 0:
                infinite_by_column[column] = infinite_count

        total_infinite = int(
            sum(infinite_by_column.values())
        )

        # ----------------------------------------------------------
        # COMPLETELY EMPTY COLUMNS
        # ----------------------------------------------------------

        empty_columns = [
            column
            for column in analysis_df.columns
            if analysis_df[column].isna().all()
        ]

        # ----------------------------------------------------------
        # CONSTANT COLUMNS
        # ----------------------------------------------------------

        constant_columns = [
            column
            for column in analysis_df.columns
            if (
                column not in empty_columns
                and analysis_df[column].nunique(dropna=True) <= 1
            )
        ]

        # ----------------------------------------------------------
        # BUILD ANALYSIS REPORT
        # ----------------------------------------------------------

        self.analysis = {
            "dataset_shape": {
                "rows": total_rows,
                "columns": total_columns
            },

            "missing_values": {
                "total": total_missing,
                "by_column": missing_columns,
                "affected_columns": len(missing_columns)
            },

            "duplicate_rows": duplicate_rows,

            "infinite_values": {
                "total": total_infinite,
                "by_column": infinite_by_column
            },

            "empty_columns": {
                "count": len(empty_columns),
                "columns": empty_columns
            },

            "constant_columns": {
                "count": len(constant_columns),
                "columns": constant_columns
            }
        }

        return self.analysis
    def _detect_numeric_like_columns(self, df, threshold=0.8):
        """
        Detect object columns that are likely intended to contain
        numeric values.

        A column is considered numeric-like when at least `threshold`
        proportion of its non-missing values can be converted to numbers.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataset to analyze.

        threshold : float, default=0.8
            Minimum proportion of valid numeric values required.

        Returns
        -------
        dict
            Dictionary containing numeric-like columns and information
            about invalid values detected in each column.
        """

        numeric_like_columns = {}

        for column in df.columns:

            # Skip columns that are already numeric
            if pd.api.types.is_numeric_dtype(df[column]):
                continue

            # Work with non-null values only
            non_null_values = df[column].dropna()

            if len(non_null_values) == 0:
                continue

            # Attempt numeric conversion
            converted_values = pd.to_numeric(
                non_null_values,
                errors="coerce"
            )

            valid_numeric_count = int(
                converted_values.notna().sum()
            )

            total_values = len(non_null_values)

            numeric_ratio = (
                valid_numeric_count / total_values
                if total_values > 0
                else 0
            )

            # If the majority of values are numeric, treat the
            # column as a numeric-like column.
            if numeric_ratio >= threshold:

                invalid_mask = converted_values.isna()

                invalid_values = (
                    non_null_values[invalid_mask]
                    .astype(str)
                    .value_counts()
                    .to_dict()
                )

                numeric_like_columns[column] = {
                    "numeric_ratio": round(
                        float(numeric_ratio),
                        4
                    ),
                    "valid_numeric_count": valid_numeric_count,
                    "invalid_value_count": int(
                        invalid_mask.sum()
                    ),
                    "invalid_values": invalid_values
                }

        return numeric_like_columns
    def _convert_invalid_numeric_values(self, df):
        """
        Convert numeric-like columns into numeric format.

        Values that cannot be converted to numbers are replaced
        with NaN so they can later be handled through imputation.
        """

        cleaned_df = df.copy()

        numeric_like_columns = self._detect_numeric_like_columns(
            cleaned_df
        )

        converted_columns = {}

        for column, info in numeric_like_columns.items():

            if info["invalid_value_count"] > 0:

                before_missing = int(
                    cleaned_df[column].isna().sum()
                )

                cleaned_df[column] = pd.to_numeric(
                    cleaned_df[column],
                    errors="coerce"
                )

                after_missing = int(
                    cleaned_df[column].isna().sum()
                )

                converted_count = after_missing - before_missing

                if converted_count > 0:
                    converted_columns[column] = converted_count

        return cleaned_df, converted_columns


    def _replace_infinite_values(self, df):
        """
        Replace positive and negative infinite values with NaN.
        """

        cleaned_df = df.copy()

        numeric_columns = cleaned_df.select_dtypes(
            include=[np.number]
        ).columns

        infinite_count = 0

        for column in numeric_columns:

            count = int(
                np.isinf(cleaned_df[column]).sum()
            )

            if count > 0:

                cleaned_df[column] = cleaned_df[column].replace(
                    [np.inf, -np.inf],
                    np.nan
                )

                infinite_count += count

        return cleaned_df, infinite_count


    def _remove_empty_columns(self, df):
        """
        Remove columns containing only missing values.
        """

        cleaned_df = df.copy()

        empty_columns = [
            column
            for column in cleaned_df.columns
            if cleaned_df[column].isna().all()
        ]

        cleaned_df = cleaned_df.drop(
            columns=empty_columns
        )

        return cleaned_df, empty_columns


    def _remove_constant_columns(self, df):
        """
        Remove columns containing only one unique non-null value.
        """

        cleaned_df = df.copy()

        constant_columns = [
            column
            for column in cleaned_df.columns
            if cleaned_df[column].nunique(dropna=True) <= 1
        ]

        cleaned_df = cleaned_df.drop(
            columns=constant_columns
        )

        return cleaned_df, constant_columns


    def _remove_duplicates(self, df):
        """
        Remove completely duplicated rows.
        """

        cleaned_df = df.copy()

        duplicate_count = int(
            cleaned_df.duplicated().sum()
        )

        if duplicate_count > 0:
            cleaned_df = cleaned_df.drop_duplicates()

        return cleaned_df, duplicate_count


    def _impute_missing_values(self, df):
        """
        Fill missing values.

        Numerical columns are filled using the median.
        Categorical columns are filled using the mode.

        If a categorical column has no valid mode, the value
        'Missing' is used.
        """

        cleaned_df = df.copy()

        imputation_report = {
            "numeric": {},
            "categorical": {}
        }

        # -----------------------------
        # NUMERICAL COLUMNS
        # -----------------------------

        numeric_columns = cleaned_df.select_dtypes(
            include=[np.number]
        ).columns

        for column in numeric_columns:

            missing_count = int(
                cleaned_df[column].isna().sum()
            )

            if missing_count > 0:

                median_value = cleaned_df[column].median()

                # If the median cannot be calculated,
                # use 0 as a final fallback.
                if pd.isna(median_value):
                    median_value = 0

                cleaned_df[column] = (
                    cleaned_df[column]
                    .fillna(median_value)
                )

                imputation_report["numeric"][column] = {
                    "count": missing_count,
                    "strategy": "median"
                }

        # -----------------------------
        # CATEGORICAL COLUMNS
        # -----------------------------

        categorical_columns = cleaned_df.select_dtypes(
            exclude=[np.number]
        ).columns

        for column in categorical_columns:

            missing_count = int(
                cleaned_df[column].isna().sum()
            )

            if missing_count > 0:

                mode_values = cleaned_df[column].mode(
                    dropna=True
                )

                if not mode_values.empty:

                    fill_value = mode_values.iloc[0]
                    strategy = "mode"

                else:

                    fill_value = "Missing"
                    strategy = "Missing"

                cleaned_df[column] = (
                    cleaned_df[column]
                    .fillna(fill_value)
                )

                imputation_report["categorical"][column] = {
                    "count": missing_count,
                    "strategy": strategy
                }

        return cleaned_df, imputation_report
    def clean_dataset(
        self,
        target_column=None,
        remove_duplicates=True,
        remove_empty_columns=True,
        remove_constant_columns=True,
        impute_missing=True
    ):
        """
        Apply the dataset cleaning pipeline.

        The original dataset is never modified. A cleaned copy is
        created and returned along with a detailed cleaning summary.

        Parameters
        ----------
        target_column : str, optional
            Name of the target column. Rows with missing target values
            are removed instead of imputing the target.

        remove_duplicates : bool, default=True
            Remove completely duplicated rows.

        remove_empty_columns : bool, default=True
            Remove columns containing only missing values.

        remove_constant_columns : bool, default=True
            Remove columns containing only one unique value.

        impute_missing : bool, default=True
            Impute missing feature values using median for numerical
            columns and mode for categorical columns.

        Returns
        -------
        tuple
            (cleaned_dataframe, cleaning_summary)
        """

        # ----------------------------------------------------------
        # START WITH A FRESH COPY OF THE ORIGINAL DATASET
        # ----------------------------------------------------------

        cleaned_df = self.original_df.copy()

        original_rows = len(cleaned_df)
        original_columns = len(cleaned_df.columns)

        # ----------------------------------------------------------
        # NORMALIZE TEXTUAL MISSING VALUES
        # ----------------------------------------------------------

        cleaned_df = self._normalize_missing_values(cleaned_df)

        normalized_missing_count = int(
            cleaned_df.isna().sum().sum()
            - self.original_df.isna().sum().sum()
        )

        # ----------------------------------------------------------
        # CONVERT INVALID NUMERIC VALUES
        # ----------------------------------------------------------

        cleaned_df, invalid_numeric_conversions = (
            self._convert_invalid_numeric_values(cleaned_df)
        )

        invalid_numeric_count = int(
            sum(invalid_numeric_conversions.values())
        )

        # ----------------------------------------------------------
        # REPLACE INFINITE VALUES
        # ----------------------------------------------------------

        cleaned_df, infinite_count = (
            self._replace_infinite_values(cleaned_df)
        )

        # ----------------------------------------------------------
        # REMOVE ROWS WITH MISSING TARGET VALUES
        # ----------------------------------------------------------

        missing_target_rows = 0

        if (
            target_column is not None
            and target_column in cleaned_df.columns
        ):

            missing_target_rows = int(
                cleaned_df[target_column].isna().sum()
            )

            if missing_target_rows > 0:

                cleaned_df = cleaned_df.dropna(
                    subset=[target_column]
                )

        # ----------------------------------------------------------
        # REMOVE COMPLETELY EMPTY COLUMNS
        # ----------------------------------------------------------

        removed_empty_columns = []

        if remove_empty_columns:

            cleaned_df, removed_empty_columns = (
                self._remove_empty_columns(cleaned_df)
            )

        # ----------------------------------------------------------
        # REMOVE CONSTANT COLUMNS
        # ----------------------------------------------------------

        removed_constant_columns = []

        if remove_constant_columns:

            # Never remove the target column as part of automatic
            # constant-column removal.
            feature_columns = [
                column
                for column in cleaned_df.columns
                if column != target_column
            ]

            constant_columns = [
                column
                for column in feature_columns
                if cleaned_df[column].nunique(
                    dropna=True
                ) <= 1
            ]

            if constant_columns:

                cleaned_df = cleaned_df.drop(
                    columns=constant_columns
                )

            removed_constant_columns = constant_columns

        # ----------------------------------------------------------
        # REMOVE DUPLICATE ROWS
        # ----------------------------------------------------------

        duplicate_rows_removed = 0

        if remove_duplicates:

            cleaned_df, duplicate_rows_removed = (
                self._remove_duplicates(cleaned_df)
            )

        # ----------------------------------------------------------
        # IMPUTE MISSING FEATURE VALUES
        # ----------------------------------------------------------

        imputation_report = {
            "numeric": {},
            "categorical": {}
        }

        if impute_missing:

            if (
                target_column is not None
                and target_column in cleaned_df.columns
            ):

                # Temporarily separate the target so that it is not
                # imputed as a normal feature.
                target_series = cleaned_df[target_column].copy()

                feature_df = cleaned_df.drop(
                    columns=[target_column]
                )

                feature_df, imputation_report = (
                    self._impute_missing_values(feature_df)
                )

                feature_df[target_column] = target_series

                # Restore the target to its original column position.
                original_column_order = [
                    column
                    for column in cleaned_df.columns
                    if column in feature_df.columns
                ]

                cleaned_df = feature_df[
                    original_column_order
                ]

            else:

                cleaned_df, imputation_report = (
                    self._impute_missing_values(cleaned_df)
                )

        # ----------------------------------------------------------
        # FINAL DATASET STATISTICS
        # ----------------------------------------------------------

        final_rows = len(cleaned_df)
        final_columns = len(cleaned_df.columns)

        remaining_missing = int(
            cleaned_df.isna().sum().sum()
        )

        # ----------------------------------------------------------
        # BUILD CLEANING SUMMARY
        # ----------------------------------------------------------

        cleaning_summary = {
            "original_dataset": {
                "rows": original_rows,
                "columns": original_columns,
                "missing_values": int(
                    self.original_df.isna().sum().sum()
                )
            },

            "cleaning_actions": {
                "normalized_missing_values": normalized_missing_count,

                "invalid_numeric_values_converted":
                    invalid_numeric_count,

                "invalid_numeric_columns":
                    invalid_numeric_conversions,

                "infinite_values_replaced":
                    infinite_count,

                "rows_removed_missing_target":
                    missing_target_rows,

                "duplicate_rows_removed":
                    duplicate_rows_removed,

                "empty_columns_removed":
                    removed_empty_columns,

                "constant_columns_removed":
                    removed_constant_columns,

                "imputation": imputation_report
            },

            "processed_dataset": {
                "rows": final_rows,
                "columns": final_columns,
                "missing_values": remaining_missing
            }
        }

        # Save the final working dataframe and log.
        self.df = cleaned_df.copy()
        self.cleaning_log = cleaning_summary

        return cleaned_df, cleaning_summary
    def get_chart_data(self, cleaning_summary=None):
        """
        Prepare before-and-after preprocessing data for visualization.

        The returned structure is JSON-friendly and can be passed
        directly to the Flask frontend.

        Parameters
        ----------
        cleaning_summary : dict, optional
            Cleaning summary returned by clean_dataset().
            If not provided, the most recent cleaning log is used.

        Returns
        -------
        dict
            Chart-ready data for preprocessing visualizations.
        """

        if cleaning_summary is None:
            cleaning_summary = self.cleaning_log

        if not cleaning_summary:
            return {
                "before_after": {
                    "labels": [],
                    "before": [],
                    "after": []
                },
                "actions": {
                    "labels": [],
                    "values": []
                }
            }

        original = cleaning_summary["original_dataset"]
        processed = cleaning_summary["processed_dataset"]
        actions = cleaning_summary["cleaning_actions"]

        # ----------------------------------------------------------
        # BEFORE VS AFTER DATA
        # ----------------------------------------------------------

        before_after = {
            "labels": [
                "Rows",
                "Columns",
                "Missing Values"
            ],

            "before": [
                original["rows"],
                original["columns"],
                original["missing_values"]
            ],

            "after": [
                processed["rows"],
                processed["columns"],
                processed["missing_values"]
            ]
        }

        # ----------------------------------------------------------
        # CLEANING ACTIONS DATA
        # ----------------------------------------------------------

        action_labels = [
            "Missing Values Normalized",
            "Invalid Numeric Values",
            "Infinite Values",
            "Rows Missing Target",
            "Duplicate Rows"
        ]

        action_values = [
            actions.get("normalized_missing_values", 0),
            actions.get("invalid_numeric_values_converted", 0),
            actions.get("infinite_values_replaced", 0),
            actions.get("rows_removed_missing_target", 0),
            actions.get("duplicate_rows_removed", 0)
        ]

        # Count removed columns
        empty_columns_removed = len(
            actions.get("empty_columns_removed", [])
        )

        constant_columns_removed = len(
            actions.get("constant_columns_removed", [])
        )

        action_labels.extend([
            "Empty Columns Removed",
            "Constant Columns Removed"
        ])

        action_values.extend([
            empty_columns_removed,
            constant_columns_removed
        ])

        # Remove actions where nothing happened.
        filtered_labels = []
        filtered_values = []

        for label, value in zip(
            action_labels,
            action_values
        ):

            if value > 0:
                filtered_labels.append(label)
                filtered_values.append(value)

        actions_chart = {
            "labels": filtered_labels,
            "values": filtered_values
        }

        # ----------------------------------------------------------
        # RETURN COMPLETE CHART DATA
        # ----------------------------------------------------------

        return {
            "before_after": before_after,
            "actions": actions_chart
        }