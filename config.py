## widget incremental path
##########################################
dbutils.widgets.text("input_file_path", "", "Input File Path")
dbutils.widgets.dropdown("trigger_flag", "False", ["True", "False"], "Trigger Flag")
dbutils.widgets.text("alternative_file_path", "abfss://histo@storage.dfs.core.windows.net/monthly_data/2026/Jan/Forecast_Jan'26.csv", "Alternative File Path")

input_file_path = dbutils.widgets.get("input_file_path")
trigger_flag = dbutils.widgets.get("trigger_flag")
alternative_path = dbutils.widgets.get("alternative_file_path")

print("input_file_path:", input_file_path)
print("trigger_flag:", trigger_flag)
print("alternative_path:", alternative_path)

if trigger_flag == "True":
    def list_files_recursive(path):
        items = dbutils.fs.ls(path)
        files = []
        for item in items:
            if item.isDir():
                if "__unitystorage" not in item.path and "models" not in item.path:
                    files.extend(list_files_recursive(item.path))
            else:
                files.append({"path": item.path, "modificationTime": item.modificationTime})
        return files
    all_files = list_files_recursive(input_file_path)
    import pandas as pd
    df_files = pd.DataFrame(all_files)
    df_files['modificationTime'] = pd.to_datetime(df_files['modificationTime'], unit='ms')
    latest_file_path = df_files.loc[df_files['modificationTime'].idxmax(), 'path']
    full_path = latest_file_path
else:
    full_path = alternative_path

display(full_path)


####################################

# =============================================================================
# DATA QUALITY ANALYZER
# =============================================================================
# Runs on raw input data immediately after file read (before transformation).
# 20 checks — ALL scoped to the 14 required columns only.
# Results saved as JSON to Azure Blob Storage:
#   container: dbnotebook-logs
#   path:      dq/{year}/{month}/dq_report_{year}_{month}.json
# =============================================================================

import pandas as pd
import numpy as np
import json
import re
from datetime import datetime
from azure.storage.blob import BlobServiceClient


class DataQualityAnalyzer:
    """
    Analyze the quality of raw input data before transformation.

    All checks are scoped to REQUIRED_COLUMNS only (14 columns).
    Columns outside this list are completely ignored.

    Results are saved as a JSON file to Azure Blob Storage.
    """

    # =====================================================================
    # CONFIGURATION — the 14 columns the pipeline needs
    # =====================================================================

    REQUIRED_COLUMNS = [
        "Revenue_Category",
        "ProbabilityPer",
        "Forecast_Jan", "Forecast_Feb", "Forecast_Mar", "Forecast_Apr",
        "Forecast_May", "Forecast_Jun", "Forecast_Jul", "Forecast_Aug",
        "Forecast_Sep", "Forecast_Oct", "Forecast_Nov", "Forecast_Dec",
    ]

    FORECAST_COLUMNS = [
        "Forecast_Jan", "Forecast_Feb", "Forecast_Mar", "Forecast_Apr",
        "Forecast_May", "Forecast_Jun", "Forecast_Jul", "Forecast_Aug",
        "Forecast_Sep", "Forecast_Oct", "Forecast_Nov", "Forecast_Dec",
    ]

    KNOWN_CATEGORIES = {
        "Committed - Signed", "Committed - Unsigned", "Wtd. Pipeline",
        "Risk", "Opportunity", "Un-id", "-",
    }

    NULL_THRESHOLD_PCT = 20.0
    EXPECTED_ROW_RANGE = (1_000, 100_000)

    # =====================================================================
    # CONSTRUCTOR
    # =====================================================================

    def __init__(self, df: pd.DataFrame, file_path: str = None):
        """
        Args:
            df: Raw DataFrame (directly from spark.read...toPandas()).
            file_path: Source file path (for extracting year/month and logging).
        """
        self.df = df
        self.file_path = file_path or "unknown"
        self.report = {}
        self.issues = []
        self.warnings = []

        # Extract year and month from file_path
        self._year, self._month = self._extract_year_month(self.file_path)

    @staticmethod
    def _extract_year_month(path: str) -> tuple:
        """
        Extract year and month from the ABFSS file path.

        Example path:
          abfss://historicaldata@gtairnistorage.dfs.core.windows.net/
          monthly_data/2026/Jan/Forecast_Jan'26.csv

        Returns:
            (year_str, month_str) e.g. ("2026", "Jan")
        """
        match = re.search(r'/(\d{4})/([A-Za-z]{3,})/', path)
        if match:
            return match.group(1), match.group(2)

        # Fallback: try to find year and month separately
        parts = path.replace("\\", "/").split("/")
        year_str, month_str = None, None
        month_names = {
            "jan", "feb", "mar", "apr", "may", "jun",
            "jul", "aug", "sep", "oct", "nov", "dec",
        }
        for i, part in enumerate(parts):
            if part.isdigit() and len(part) == 4:
                year_str = part
                if i + 1 < len(parts) and parts[i + 1].lower()[:3] in month_names:
                    month_str = parts[i + 1]
                break

        return year_str or "unknown", month_str or "unknown"

    def _log(self, msg: str):
        """Print to notebook output only (no logger)."""
        print(msg)

    # =================================================================
    # JSON EXPORT — save report to Azure Blob Storage
    # =================================================================
    def save_report_to_blob(
        self,
        connection_string: str,
        container_name: str = "dbnotebook-logs",
        blob_path_prefix: str = "dq",
        filename: str = None,
    ) -> str:
        """
        Save the DQ report as a JSON file to Azure Blob Storage.

        Blob path: {blob_path_prefix}/{year}/{month}/{filename}.json
        Example:   dq/2026/Jan/dq_report_2026_Jan.json

        Args:
            connection_string: Azure Storage connection string.
            container_name: Blob container name.
            blob_path_prefix: Folder prefix in the container.
            filename: Custom filename. If None, auto-generated.

        Returns:
            Full blob path of the saved JSON file.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dq_report_{self._year}_{self._month}_{timestamp}.json"

        if not filename.endswith(".json"):
            filename += ".json"

        blob_path = f"{blob_path_prefix}/{self._year}/{self._month}/{filename}"

        # Convert report to JSON-serializable format
        json_report = self._make_json_serializable(self.report)

        # Add metadata
        json_report["_metadata"] = {
            "file_path": self.file_path,
            "year": self._year,
            "month": self._month,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_checks": 20,
            "total_issues": len(self.issues),
            "total_warnings": len(self.warnings),
            "issues": self.issues,
            "warnings": self.warnings,
        }

        # Upload to Azure Blob Storage
        json_str = json.dumps(json_report, indent=2, ensure_ascii=False)

        blob_service_client = BlobServiceClient.from_connection_string(
            connection_string
        )
        blob_client = blob_service_client.get_blob_client(
            container=container_name, blob=blob_path
        )
        blob_client.upload_blob(
            json_str.encode("utf-8"), overwrite=True
        )

        full_url = (
            f"wasbs://{container_name}@gtairnistorage.blob.core.windows.net/"
            f"{blob_path}"
        )
        self._log(f"\nDQ Report saved to: {full_url}")

        return blob_path

    @staticmethod
    def _make_json_serializable(obj):
        """Recursively convert numpy/pandas types to Python native types."""
        if isinstance(obj, dict):
            return {
                k: DataQualityAnalyzer._make_json_serializable(v)
                for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [
                DataQualityAnalyzer._make_json_serializable(v) for v in obj
            ]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, set):
            return sorted(list(obj))
        else:
            return obj

    # =================================================================
    # CHECK  1 — FILE & SHAPE INFO
    # =================================================================
    def check_shape(self) -> dict:
        """Report DataFrame shape, file path, year/month, timestamp."""
        n_rows, n_cols = self.df.shape
        req_present = [
            c for c in self.REQUIRED_COLUMNS if c in self.df.columns
        ]

        info = {
            "total_rows": n_rows,
            "total_columns": n_cols,
            "required_columns_present": len(req_present),
            "required_columns_total": len(self.REQUIRED_COLUMNS),
            "file_path": self.file_path,
            "extracted_year": self._year,
            "extracted_month": self._month,
            "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.report["shape"] = info

        self._log(f"\n{'='*70}")
        self._log(f"DATA QUALITY ANALYZER")
        self._log(f"{'='*70}")
        self._log(f"File Path        : {self.file_path}")
        self._log(f"Year / Month     : {self._year} / {self._month}")
        self._log(f"Analysis Time    : {info['analysis_timestamp']}")
        self._log(f"Total Rows       : {n_rows:,}")
        self._log(
            f"Total Columns    : {n_cols} "
            f"(checking {len(req_present)} required)"
        )

        if n_rows == 0:
            self.issues.append("CRITICAL: DataFrame has 0 rows — empty file!")

        return info

    # =================================================================
    # CHECK  2 — REQUIRED COLUMN CHECK (Column Signature)
    # =================================================================
    def check_required_columns(self, extra_required: list = None) -> dict:
        """Check if all 14 required columns are present."""
        required = list(self.REQUIRED_COLUMNS)
        if extra_required:
            required.extend(extra_required)

        actual_cols = set(self.df.columns)
        present = [c for c in required if c in actual_cols]
        missing = [c for c in required if c not in actual_cols]

        info = {
            "required_count": len(required),
            "present_count": len(present),
            "missing_count": len(missing),
            "missing_columns": missing,
            "present_columns": present,
        }
        self.report["column_signature"] = info

        self._log(f"\n--- Column Signature (14 Required Columns) ---")
        self._log(f"Required         : {len(required)}")
        self._log(f"Present          : {len(present)}")
        self._log(f"Missing          : {len(missing)}")

        if missing:
            self.issues.append(
                f"CRITICAL: Missing required columns: {missing}"
            )
            self._log(f"  MISSING: {missing}")
        else:
            self._log(f"  All 14 required columns present.")

        return info

    # =================================================================
    # CHECK  3 — NULL VALUE ANALYSIS (required columns only)
    # =================================================================
    def check_nulls(self) -> pd.DataFrame:
        """Check null values for required columns only."""
        cols_to_check = [
            c for c in self.REQUIRED_COLUMNS if c in self.df.columns
        ]

        if not cols_to_check:
            self._log(f"\n--- Null Value Analysis ---")
            self._log(f"  No required columns found to check.")
            return pd.DataFrame()

        null_stats = []
        for col in cols_to_check:
            total = len(self.df)
            null_count = self.df[col].isnull().sum()
            null_pct = (null_count / total * 100) if total > 0 else 0
            status = "OK" if null_pct < self.NULL_THRESHOLD_PCT else "BAD"

            null_stats.append({
                "column": col,
                "total_rows": total,
                "null_count": int(null_count),
                "null_pct": round(null_pct, 2),
                "status": status,
            })

            if null_pct >= self.NULL_THRESHOLD_PCT:
                self.issues.append(
                    f"Column '{col}' has {null_pct:.1f}% nulls "
                    f"(threshold: {self.NULL_THRESHOLD_PCT}%)"
                )

        null_df = pd.DataFrame(null_stats)
        self.report["null_analysis"] = null_df.to_dict(orient="records")

        self._log(
            f"\n--- Null Value Analysis "
            f"(threshold: {self.NULL_THRESHOLD_PCT}%) ---"
        )
        self._log(f"Columns checked  : {len(cols_to_check)} (required only)")
        bad_cols = null_df[null_df["status"] == "BAD"]
        ok_cols = null_df[null_df["status"] == "OK"]
        self._log(f"Columns OK       : {len(ok_cols)}")
        self._log(f"Columns FLAGGED  : {len(bad_cols)}")

        if len(bad_cols) > 0:
            self._log(
                f"\n  Flagged columns "
                f"(null % >= {self.NULL_THRESHOLD_PCT}%):"
            )
            for _, row in bad_cols.iterrows():
                self._log(
                    f"    {row['column']:<35} -> "
                    f"{row['null_pct']:>6.1f}% nulls "
                    f"({row['null_count']:,} / {row['total_rows']:,})"
                )

        self._log(f"\n  Full null summary (required columns):")
        for _, row in null_df.iterrows():
            marker = "!!" if row["status"] == "BAD" else "  "
            self._log(
                f"  {marker} {row['column']:<35} "
                f"{row['null_pct']:>6.1f}%  ({row['null_count']:,} nulls)"
            )

        return null_df

    # =================================================================
    # CHECK  4 — DATA TYPES CHECK (required columns only)
    # =================================================================
    def check_dtypes(self) -> dict:
        """Check data types of required columns; flag mismatches."""
        cols_to_check = [
            c for c in self.REQUIRED_COLUMNS if c in self.df.columns
        ]

        dtype_info = {}
        type_issues = []

        for col in cols_to_check:
            dtype = str(self.df[col].dtype)
            dtype_info[col] = dtype

            if col.startswith("Forecast_") and dtype == "object":
                type_issues.append(
                    f"'{col}' is type '{dtype}' — expected numeric"
                )
            if col == "ProbabilityPer" and dtype == "object":
                type_issues.append(
                    f"'{col}' is type '{dtype}' — expected numeric"
                )

        self.report["dtypes"] = dtype_info

        self._log(f"\n--- Data Types (required columns only) ---")
        for col, dtype in dtype_info.items():
            self._log(f"  {col:<35} {dtype}")

        if type_issues:
            for issue in type_issues:
                self.warnings.append(f"Data type warning: {issue}")
                self._log(f"  WARNING: {issue}")

        return dtype_info

    # =================================================================
    # CHECK  5 — ZERO FORECAST ROW DETECTION
    # =================================================================
    def check_zero_forecast_rows(self) -> dict:
        """Detect rows where ALL 12 forecast columns are zero."""
        forecast_cols = [
            c for c in self.FORECAST_COLUMNS if c in self.df.columns
        ]

        if not forecast_cols:
            self._log(f"\n--- Zero Forecast Rows ---")
            self._log(f"  No forecast columns found, skipping.")
            return {"zero_forecast_rows": 0}

        forecast_data = self.df[forecast_cols].apply(
            pd.to_numeric, errors="coerce"
        ).fillna(0)

        all_zero_mask = (forecast_data == 0).all(axis=1)
        zero_count = int(all_zero_mask.sum())
        zero_pct = (
            (zero_count / len(self.df) * 100) if len(self.df) > 0 else 0
        )

        info = {
            "zero_forecast_rows": zero_count,
            "zero_forecast_pct": round(zero_pct, 2),
            "total_rows": len(self.df),
            "forecast_columns_checked": len(forecast_cols),
        }
        self.report["zero_forecast"] = info

        self._log(f"\n--- Zero Forecast Rows ---")
        self._log(f"Forecast columns checked : {len(forecast_cols)}")
        self._log(
            f"Rows with ALL zeros      : {zero_count:,} ({zero_pct:.1f}%)"
        )

        if zero_count > 0:
            self.warnings.append(
                f"{zero_count:,} rows ({zero_pct:.1f}%) "
                f"have all-zero forecast columns"
            )
            if "Revenue_Category" in self.df.columns:
                zero_rows = self.df[all_zero_mask]
                cat_counts = zero_rows["Revenue_Category"].value_counts()
                self._log(f"  Breakdown by Revenue_Category:")
                for cat, cnt in cat_counts.items():
                    self._log(f"    {str(cat):<30} -> {cnt:,} rows")

        return info

    # =================================================================
    # CHECK  6 — REVENUE CATEGORY DISTRIBUTION
    # =================================================================
    def check_revenue_categories(self) -> dict:
        """Analyze Revenue_Category value distribution."""
        col = "Revenue_Category"
        if col not in self.df.columns:
            self._log(f"\n--- Revenue Category Distribution ---")
            self._log(f"  Column '{col}' not found, skipping.")
            return {}

        cat_counts = self.df[col].value_counts(dropna=False)
        total = len(self.df)

        info = {
            "unique_categories": int(cat_counts.count()),
            "distribution": {str(k): int(v) for k, v in cat_counts.items()},
        }
        self.report["revenue_category"] = info

        self._log(f"\n--- Revenue Category Distribution ---")
        self._log(f"Unique categories: {info['unique_categories']}")
        for cat, cnt in cat_counts.items():
            pct = cnt / total * 100
            self._log(
                f"  {str(cat):<35} {cnt:>8,} rows  ({pct:>5.1f}%)"
            )

        return info

    # =================================================================
    # CHECK  7 — DESCRIPTIVE STATISTICS (required numeric columns only)
    # =================================================================
    def check_statistics(self) -> pd.DataFrame:
        """Run .describe() on forecast columns + ProbabilityPer."""
        forecast_cols = [
            c for c in self.FORECAST_COLUMNS if c in self.df.columns
        ]
        num_cols = forecast_cols + (
            ["ProbabilityPer"]
            if "ProbabilityPer" in self.df.columns
            else []
        )

        if not num_cols:
            self._log(f"\n--- Descriptive Statistics ---")
            self._log(f"  No numeric required columns to describe.")
            return pd.DataFrame()

        desc = self.df[num_cols].apply(
            pd.to_numeric, errors="coerce"
        ).describe()

        # Convert describe output to JSON-safe dict
        self.report["statistics"] = {
            col: {
                stat: round(float(val), 4)
                for stat, val in desc[col].items()
            }
            for col in desc.columns
        }

        self._log(f"\n--- Descriptive Statistics (required columns only) ---")
        self._log(f"{desc.to_string()}")

        return desc

    # =================================================================
    # CHECK  8 — SAMPLE DATA (required columns only)
    # =================================================================
    def show_sample(self, n: int = 5) -> pd.DataFrame:
        """Show sample rows (first N) for required columns only."""
        cols_to_show = [
            c for c in self.REQUIRED_COLUMNS if c in self.df.columns
        ]

        if not cols_to_show:
            self._log(f"\n--- Sample Data ---")
            self._log(f"  No required columns found to display.")
            return pd.DataFrame()

        sample = self.df[cols_to_show].head(n)

        # Store as JSON-safe records
        self.report["sample_data"] = sample.fillna("null").to_dict(
            orient="records"
        )

        self._log(
            f"\n--- Sample Data "
            f"(first {n} rows, {len(cols_to_show)} required columns) ---"
        )
        self._log(f"{sample.to_string()}")

        return sample

    # =================================================================
    # CHECK 10 — DUPLICATE ROWS (based on required columns only)
    # =================================================================
    def check_duplicates(self) -> dict:
        """Detect duplicate rows based on 14 required columns only."""
        cols_to_check = [
            c for c in self.REQUIRED_COLUMNS if c in self.df.columns
        ]

        if not cols_to_check:
            return {"duplicate_rows": 0, "duplicate_pct": 0}

        total = len(self.df)
        dup_count = int(self.df.duplicated(subset=cols_to_check).sum())
        dup_pct = (dup_count / total * 100) if total > 0 else 0

        info = {
            "duplicate_rows": dup_count,
            "duplicate_pct": round(dup_pct, 2),
            "columns_checked": len(cols_to_check),
        }
        self.report["duplicates"] = info

        self._log(
            f"\n--- Duplicate Rows "
            f"(based on {len(cols_to_check)} required columns) ---"
        )
        self._log(f"Exact duplicates : {dup_count:,} ({dup_pct:.1f}%)")

        if dup_count > 0:
            self.warnings.append(
                f"{dup_count:,} duplicate rows found ({dup_pct:.1f}%)"
            )
            if "Revenue_Category" in self.df.columns:
                dup_rows = self.df[
                    self.df.duplicated(subset=cols_to_check, keep=False)
                ]
                cat_counts = (
                    dup_rows["Revenue_Category"].value_counts().head(5)
                )
                self._log(f"  Top categories with duplicates:")
                for cat, cnt in cat_counts.items():
                    self._log(f"    {str(cat):<30} -> {cnt:,} rows")

        return info

    # =================================================================
    # CHECK 11 — ROW COUNT RANGE VALIDATION
    # =================================================================
    def check_row_count_range(self) -> dict:
        """Validate row count is within expected range (30k-60k)."""
        total = len(self.df)
        lo, hi = self.EXPECTED_ROW_RANGE
        in_range = lo <= total <= hi

        info = {
            "total_rows": total,
            "expected_range": f"{lo:,} - {hi:,}",
            "in_range": in_range,
        }
        self.report["row_count_range"] = info

        self._log(f"\n--- Row Count Range ---")
        self._log(f"Total rows       : {total:,}")
        self._log(f"Expected range   : {lo:,} - {hi:,}")

        if not in_range:
            msg = (
                f"Row count {total:,} is OUTSIDE expected range "
                f"({lo:,} - {hi:,})"
            )
            self.warnings.append(msg)
            self._log(f"  !! {msg}")
        else:
            self._log(f"  Row count is within expected range.")

        return info

    # =================================================================
    # CHECK 12 — COLUMN NAME WHITESPACE / CASE MISMATCH (required only)
    # =================================================================
    def check_column_name_issues(self) -> dict:
        """Detect whitespace or case mismatches for required columns only."""
        actual_cols = list(self.df.columns)
        issues_found = []

        actual_lower_map = {c.lower().strip(): c for c in actual_cols}

        for req_col in self.REQUIRED_COLUMNS:
            if req_col in actual_cols:
                continue

            match = actual_lower_map.get(req_col.lower().strip())
            if match:
                if match != match.strip():
                    issues_found.append(
                        f"Whitespace in column: found '{match}', "
                        f"expected '{req_col}'"
                    )
                    self.issues.append(
                        f"Column name has whitespace: '{match}' "
                        f"should be '{req_col}'"
                    )
                else:
                    issues_found.append(
                        f"Case mismatch: expected '{req_col}', "
                        f"found '{match}'"
                    )
                    self.warnings.append(
                        f"Column case mismatch: '{match}' "
                        f"should be '{req_col}'"
                    )

        info = {"issues": issues_found, "count": len(issues_found)}
        self.report["column_name_issues"] = info

        self._log(f"\n--- Column Name Issues (required columns only) ---")
        if issues_found:
            for issue in issues_found:
                self._log(f"  !! {issue}")
        else:
            self._log(f"  No column name issues detected.")

        return info

    # =================================================================
    # CHECK 13 — NEGATIVE FORECAST VALUES
    # =================================================================
    def check_negative_forecasts(self) -> dict:
        """Flag rows with negative values in forecast columns."""
        forecast_cols = [
            c for c in self.FORECAST_COLUMNS if c in self.df.columns
        ]
        if not forecast_cols:
            return {}

        forecast_data = self.df[forecast_cols].apply(
            pd.to_numeric, errors="coerce"
        )
        negative_mask = forecast_data < 0

        neg_counts = {}
        total_negative_cells = 0
        for col in forecast_cols:
            neg_count = int(negative_mask[col].sum())
            if neg_count > 0:
                neg_counts[col] = neg_count
                total_negative_cells += neg_count

        info = {
            "columns_with_negatives": neg_counts,
            "total_negative_cells": total_negative_cells,
        }
        self.report["negative_forecasts"] = info

        self._log(f"\n--- Negative Forecast Values ---")
        if neg_counts:
            self.warnings.append(
                f"{total_negative_cells} negative values found "
                f"across forecast columns"
            )
            for col, cnt in neg_counts.items():
                self._log(f"  {col:<25} -> {cnt:,} negative values")
        else:
            self._log(f"  No negative forecast values found.")

        return info

    # =================================================================
    # CHECK 14 — INFINITY VALUES (required numeric columns only)
    # =================================================================
    def check_infinity_values(self) -> dict:
        """Detect inf/-inf in forecast columns + ProbabilityPer only."""
        forecast_cols = [
            c for c in self.FORECAST_COLUMNS if c in self.df.columns
        ]
        check_cols = forecast_cols + (
            ["ProbabilityPer"]
            if "ProbabilityPer" in self.df.columns
            else []
        )

        if not check_cols:
            return {}

        inf_counts = {}
        for col in check_cols:
            numeric_col = pd.to_numeric(self.df[col], errors="coerce")
            inf_count = int(np.isinf(numeric_col).sum())
            if inf_count > 0:
                inf_counts[col] = inf_count

        info = {"columns_with_inf": inf_counts}
        self.report["infinity_values"] = info

        self._log(f"\n--- Infinity Values (required columns only) ---")
        if inf_counts:
            self.issues.append(f"Infinity values found: {inf_counts}")
            for col, cnt in inf_counts.items():
                self._log(f"  !! {col:<25} -> {cnt:,} inf values")
        else:
            self._log(f"  No infinity values detected.")

        return info

    # =================================================================
    # CHECK 15 — PROBABILITY PERCENTAGE RANGE
    # =================================================================
    def check_probability_range(self) -> dict:
        """Flag ProbabilityPer values outside [0, 100]."""
        col = "ProbabilityPer"
        if col not in self.df.columns:
            return {}

        prob = pd.to_numeric(self.df[col], errors="coerce")
        valid = prob.dropna()

        if len(valid) == 0:
            self._log(f"\n--- ProbabilityPer Range Check ---")
            self._log(f"  All values are null, skipping.")
            return {"total_non_null": 0}

        below_zero = int((valid < 0).sum())
        above_100 = int((valid > 100).sum())
        in_decimal_range = int(((valid >= 0) & (valid <= 1)).sum())
        in_pct_range = int(((valid > 1) & (valid <= 100)).sum())

        info = {
            "total_non_null": len(valid),
            "below_zero": below_zero,
            "above_100": above_100,
            "in_decimal_range_0_1": in_decimal_range,
            "in_percent_range_1_100": in_pct_range,
            "min": float(valid.min()),
            "max": float(valid.max()),
        }
        self.report["probability_range"] = info

        self._log(f"\n--- ProbabilityPer Range Check ---")
        self._log(f"Range            : [{info['min']}, {info['max']}]")
        self._log(f"In decimal (0-1) : {in_decimal_range:,}")
        self._log(f"In percent (1-100): {in_pct_range:,}")

        if below_zero > 0:
            self.issues.append(
                f"ProbabilityPer has {below_zero} values below 0"
            )
            self._log(f"  !! {below_zero:,} values BELOW 0")
        if above_100 > 0:
            self.issues.append(
                f"ProbabilityPer has {above_100} values above 100"
            )
            self._log(f"  !! {above_100:,} values ABOVE 100")
        if below_zero == 0 and above_100 == 0:
            self._log(f"  All values within valid range.")

        return info

    # =================================================================
    # CHECK 16 — UNEXPECTED REVENUE CATEGORY VALUES
    # =================================================================
    def check_unexpected_categories(self) -> dict:
        """Flag Revenue_Category values not in the known set."""
        col = "Revenue_Category"
        if col not in self.df.columns:
            return {}

        actual_values = set(self.df[col].dropna().unique())
        unknown = actual_values - self.KNOWN_CATEGORIES

        info = {
            "known_categories": sorted(self.KNOWN_CATEGORIES),
            "unknown_values": sorted(unknown) if unknown else [],
        }
        self.report["unexpected_categories"] = info

        self._log(f"\n--- Revenue_Category Validation ---")
        self._log(f"Known categories : {len(self.KNOWN_CATEGORIES)}")
        self._log(f"Found in data    : {len(actual_values)}")

        if unknown:
            self.warnings.append(
                f"Unknown Revenue_Category values: {sorted(unknown)}"
            )
            self._log(f"  !! Unknown values: {sorted(unknown)}")
        else:
            self._log(f"  All categories are recognized.")

        return info

    # =================================================================
    # CHECK 17 — WHITESPACE IN CATEGORICAL VALUES
    # =================================================================
    def check_categorical_whitespace(self) -> dict:
        """Detect leading/trailing whitespace in Revenue_Category."""
        col = "Revenue_Category"
        if col not in self.df.columns:
            return {}

        str_vals = self.df[col].dropna().astype(str)
        has_whitespace = str_vals != str_vals.str.strip()
        ws_count = int(has_whitespace.sum())

        whitespace_issues = {}
        if ws_count > 0:
            whitespace_issues[col] = ws_count
            examples = str_vals[has_whitespace].unique()[:3]
            self.warnings.append(
                f"Column '{col}' has {ws_count} values with whitespace, "
                f"e.g.: {list(examples)}"
            )

        info = {"columns_with_whitespace": whitespace_issues}
        self.report["categorical_whitespace"] = info

        self._log(
            f"\n--- Categorical Value Whitespace (Revenue_Category) ---"
        )
        if whitespace_issues:
            self._log(
                f"  !! Revenue_Category -> "
                f"{ws_count:,} values with leading/trailing whitespace"
            )
        else:
            self._log(f"  No whitespace issues in Revenue_Category.")

        return info

    # =================================================================
    # CHECK 18 — DEAD COLUMNS (100% NULL) — required columns only
    # =================================================================
    def check_dead_columns(self) -> dict:
        """Find required columns that are entirely null."""
        cols_to_check = [
            c for c in self.REQUIRED_COLUMNS if c in self.df.columns
        ]

        if not cols_to_check:
            self._log(f"\n--- Dead Columns (100% null) ---")
            self._log(f"  No required columns found to check.")
            return {"dead_columns": [], "count": 0}

        subset = self.df[cols_to_check]
        null_pcts = subset.isnull().mean() * 100
        dead_cols = list(null_pcts[null_pcts == 100].index)

        info = {
            "dead_columns": dead_cols,
            "count": len(dead_cols),
            "columns_checked": len(cols_to_check),
        }
        self.report["dead_columns"] = info

        self._log(
            f"\n--- Dead Columns (100% null, required columns only) ---"
        )
        self._log(
            f"Columns checked  : {len(cols_to_check)} (required only)"
        )
        self._log(f"Dead columns     : {len(dead_cols)}")

        if dead_cols:
            self.issues.append(
                f"CRITICAL: Required columns are 100% null: {dead_cols}"
            )
            self._log(f"  !! Dead REQUIRED columns: {dead_cols}")
        else:
            self._log(f"  No dead columns among required columns.")

        return info

    # =================================================================
    # CHECK 19 — OUTLIER DETECTION (IQR) — forecast columns only
    # =================================================================
    def check_outliers(self, iqr_multiplier: float = 3.0) -> dict:
        """Detect extreme outliers in forecast columns using IQR."""
        forecast_cols = [
            c for c in self.FORECAST_COLUMNS if c in self.df.columns
        ]
        if not forecast_cols:
            return {}

        outlier_info = {}
        for col in forecast_cols:
            numeric = pd.to_numeric(
                self.df[col], errors="coerce"
            ).dropna()
            if len(numeric) < 4:
                continue

            q1 = numeric.quantile(0.25)
            q3 = numeric.quantile(0.75)
            iqr = q3 - q1

            if iqr == 0:
                continue

            lower_bound = q1 - iqr_multiplier * iqr
            upper_bound = q3 + iqr_multiplier * iqr

            outlier_count = int(
                ((numeric < lower_bound) | (numeric > upper_bound)).sum()
            )
            if outlier_count > 0:
                outlier_info[col] = {
                    "count": outlier_count,
                    "bounds": f"[{lower_bound:,.0f}, {upper_bound:,.0f}]",
                    "min_value": float(numeric.min()),
                    "max_value": float(numeric.max()),
                }

        info = {
            "outlier_columns": outlier_info,
            "iqr_multiplier": iqr_multiplier,
        }
        self.report["outliers"] = info

        self._log(
            f"\n--- Outlier Detection "
            f"(IQR x {iqr_multiplier}, forecast columns) ---"
        )
        if outlier_info:
            for col, details in outlier_info.items():
                self._log(
                    f"  {col:<25} -> {details['count']:,} outliers "
                    f"outside {details['bounds']}"
                )
            self.warnings.append(
                f"Outliers found in {len(outlier_info)} forecast columns"
            )
        else:
            self._log(f"  No extreme outliers detected.")

        return info

    # =================================================================
    # CHECK 20 — ZERO-VARIANCE COLUMNS (required columns only)
    # =================================================================
    def check_zero_variance(self) -> dict:
        """Detect required columns where every value is identical."""
        cols_to_check = [
            c for c in self.REQUIRED_COLUMNS if c in self.df.columns
        ]
        zero_var_cols = []

        for col in cols_to_check:
            nunique = self.df[col].nunique(dropna=True)
            if nunique <= 1:
                zero_var_cols.append(col)

        info = {
            "zero_variance_columns": zero_var_cols,
            "count": len(zero_var_cols),
        }
        self.report["zero_variance"] = info

        self._log(
            f"\n--- Zero-Variance Columns (required columns only) ---"
        )
        if zero_var_cols:
            self.warnings.append(
                f"Zero-variance columns: {zero_var_cols}"
            )
            for col in zero_var_cols:
                val = self.df[col].dropna().unique()
                self._log(f"  !! {col:<25} -> only value: {val}")
        else:
            self._log(f"  No zero-variance columns.")

        return info

    # =================================================================
    # CHECK  9 — HEALTH SCORE (runs last — aggregates all findings)
    # =================================================================
    def compute_health_score(self) -> dict:
        """
        Compute overall data quality health score (0-100).

        Scoring breakdown:
          - Required columns present:     20 pts
          - Null % within threshold:      20 pts
          - No critical type issues:       5 pts
          - Zero-forecast rows < 50%:     10 pts
          - Row count > 0:                10 pts
          - No duplicates (< 5%):          5 pts
          - Row count in expected range:   5 pts
          - No column name issues:         5 pts
          - No infinity values:            5 pts
          - ProbabilityPer in range:       5 pts
          - No dead required columns:      5 pts
          - No zero-variance required:     5 pts
                                    Total: 100 pts
        """
        breakdown = {}

        # 1. Required columns (20 pts)
        col_info = self.report.get("column_signature", {})
        if col_info.get("missing_count", 1) == 0:
            breakdown["required_columns"] = 20
        else:
            ratio = col_info.get("missing_count", 0) / max(
                col_info.get("required_count", 1), 1
            )
            breakdown["required_columns"] = int(20 * (1 - ratio))

        # 2. Null quality (20 pts)
        null_records = self.report.get("null_analysis", [])
        if null_records:
            ok = sum(1 for r in null_records if r["status"] == "OK")
            breakdown["null_quality"] = int(
                20 * ok / len(null_records)
            )
        else:
            breakdown["null_quality"] = 20

        # 3. Data types (5 pts)
        type_warns = [w for w in self.warnings if "Data type" in w]
        breakdown["data_types"] = 5 if not type_warns else 0

        # 4. Zero forecast rows (10 pts)
        zero_pct = self.report.get("zero_forecast", {}).get(
            "zero_forecast_pct", 0
        )
        breakdown["zero_forecast"] = int(
            10 * max(0, 1 - zero_pct / 100)
        )

        # 5. Row count > 0 (10 pts)
        breakdown["row_count"] = (
            10
            if self.report.get("shape", {}).get("total_rows", 0) > 0
            else 0
        )

        # 6. Duplicates < 5% (5 pts)
        dup_pct = self.report.get("duplicates", {}).get(
            "duplicate_pct", 0
        )
        breakdown["no_duplicates"] = (
            5 if dup_pct < 5 else int(5 * (1 - dup_pct / 100))
        )

        # 7. Row count in range (5 pts)
        breakdown["row_range"] = (
            5
            if self.report.get("row_count_range", {}).get(
                "in_range", True
            )
            else 0
        )

        # 8. Column name cleanliness (5 pts)
        col_issues = self.report.get("column_name_issues", {}).get(
            "count", 0
        )
        breakdown["column_names_clean"] = 5 if col_issues == 0 else 0

        # 9. No infinity (5 pts)
        inf_cols = self.report.get("infinity_values", {}).get(
            "columns_with_inf", {}
        )
        breakdown["no_infinity"] = 5 if not inf_cols else 0

        # 10. Probability in range (5 pts)
        prob = self.report.get("probability_range", {})
        breakdown["probability_valid"] = (
            5
            if prob.get("below_zero", 0) == 0
            and prob.get("above_100", 0) == 0
            else 0
        )

        # 11. No dead required columns (5 pts)
        dead_req = self.report.get("dead_columns", {}).get(
            "dead_columns", []
        )
        breakdown["no_dead_required"] = 5 if not dead_req else 0

        # 12. No zero-variance required (5 pts)
        zv = self.report.get("zero_variance", {}).get(
            "zero_variance_columns", []
        )
        breakdown["no_zero_variance"] = 5 if not zv else 0

        score = sum(breakdown.values())
        max_score = 100

        if score >= 80:
            verdict = "HEALTHY"
        elif score >= 50:
            verdict = "WARNING — review flagged issues before proceeding"
        else:
            verdict = (
                "POOR — input data unreliable, "
                "predictions will be garbage"
            )

        health = {
            "score": score,
            "max_score": max_score,
            "verdict": verdict,
            "breakdown": breakdown,
            "total_issues": len(self.issues),
            "total_warnings": len(self.warnings),
        }
        self.report["health_score"] = health

        self._log(f"\n{'='*70}")
        self._log(
            f"DATA QUALITY HEALTH SCORE: "
            f"{score} / {max_score}  ->  {verdict}"
        )
        self._log(f"{'='*70}")
        self._log(f"Breakdown:")
        for k, v in breakdown.items():
            self._log(f"  {k:<25} {v:>3} pts")

        if self.issues:
            self._log(f"\nCritical Issues ({len(self.issues)}):")
            for i, issue in enumerate(self.issues, 1):
                self._log(f"  {i}. {issue}")

        if self.warnings:
            self._log(f"\nWarnings ({len(self.warnings)}):")
            for i, warn in enumerate(self.warnings, 1):
                self._log(f"  {i}. {warn}")

        self._log(f"{'='*70}\n")

        return health

    # =================================================================
    # MAIN — RUN ALL 20 CHECKS
    # =================================================================
    def run_full_analysis(
        self,
        extra_required_columns: list = None,
        sample_rows: int = 5,
    ) -> dict:
        """
        Run the complete data quality analysis pipeline (20 checks).
        ALL checks scoped to REQUIRED_COLUMNS (14 columns) only.

        Args:
            extra_required_columns: Additional columns beyond default 14.
            sample_rows: Number of sample rows to display.

        Returns:
            dict with full report (also accessible via self.report).
        """
        self.check_shape()                                                  #  1
        self.check_required_columns(extra_required=extra_required_columns)  #  2
        self.check_nulls()                                                  #  3
        self.check_dtypes()                                                 #  4
        self.check_zero_forecast_rows()                                     #  5
        self.check_revenue_categories()                                     #  6
        self.check_statistics()                                             #  7
        self.show_sample(n=sample_rows)                                     #  8
        self.check_duplicates()                                             # 10
        self.check_row_count_range()                                        # 11
        self.check_column_name_issues()                                     # 12
        self.check_negative_forecasts()                                     # 13
        self.check_infinity_values()                                        # 14
        self.check_probability_range()                                      # 15
        self.check_unexpected_categories()                                  # 16
        self.check_categorical_whitespace()                                 # 17
        self.check_dead_columns()                                           # 18
        self.check_outliers()                                               # 19
        self.check_zero_variance()                                          # 20
        self.compute_health_score()                                         #  9

        return self.report


# =========================================================================
# USAGE
# =========================================================================

# Read raw data from ABFSS path
raw_df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(full_path)
    .toPandas()
)

# Run Data Quality Analyzer (prints to notebook only, no logger)
analyzer = DataQualityAnalyzer(df=raw_df, file_path=full_path)

dq_report = analyzer.run_full_analysis(
    extra_required_columns=None,
    sample_rows=5,
)

# Save report as JSON to Azure Blob Storage
# Path: dbnotebook-logs/dq/{year}/{month}/dq_report_{year}_{month}_{timestamp}.json
blob_path = analyzer.save_report_to_blob(
    connection_string=connection_string,   # from "Logging" cell
    container_name="dbnotebook-logs",
    blob_path_prefix="dq",
)

print(f"\nJSON report saved at: {blob_path}")

# Optional: halt pipeline if quality is poor
if dq_report["health_score"]["score"] < 50:
    print(
        f"DATA QUALITY CHECK FAILED — "
        f"score: {dq_report['health_score']['score']}/100"
    )
    # Uncomment below to stop the pipeline:
    # raise ValueError(
    #     f"Data quality too poor to proceed "
    #     f"(score: {dq_report['health_score']['score']}/100)"
    # )


#########################################################################
#Transformation script on incremental data

import pandas as pd
import numpy as np
from typing import Optional

# Databricks-specific imports (available in Databricks runtime)
try:
    from pyspark.sql import SparkSession
    DATABRICKS_ENV = True
except ImportError:
    DATABRICKS_ENV = False


class DataTransformer:
    """
    Object-oriented class for fetching and transforming FP&A data.
    
    Modified for Databricks to work with Azure Blob Storage (ABFSS paths).
    
    Handles:
    - Fetching historical transformed data from ABFSS path
    - Fetching newly arrived raw data from ABFSS path
    """
    
    # Use __slots__ for memory optimization
    __slots__ = (
        'historical_data_path', 'raw_data_path', 
        '_historical_data', '_raw_data', '_refined_data',
        '_forecast_col_cache', '_historical_index', 'verbose',
        '_spark', '_current_year', '_current_month'
    )
    
    # Month name to number mapping (case-insensitive)
    MONTH_MAP = {
        'jan': 1, 'january': 1,
        'feb': 2, 'february': 2,
        'mar': 3, 'march': 3,
        'apr': 4, 'april': 4,
        'may': 5,
        'jun': 6, 'june': 6,
        'jul': 7, 'july': 7,
        'aug': 8, 'august': 8,
        'sep': 9, 'sept': 9, 'september': 9,
        'oct': 10, 'october': 10,
        'nov': 11, 'november': 11,
        'dec': 12, 'december': 12,
    }

    # Month number to abbreviation mapping for column lookup
    MONTH_ABBREV = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
        7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
    }
    
    # Pre-computed lowercase abbreviations for faster matching
    MONTH_ABBREV_LOWER = {k: v.lower() for k, v in MONTH_ABBREV.items()}
    
    # -------------------------------------------------------------------------
    # Constructor: Initialize DataTransformer with ABFSS paths for Databricks
    # -------------------------------------------------------------------------
    def __init__(
        self, 
        raw_data_path: str = None,
        historical_data_path: str = None,
        current_year: int = None,
        current_month: int = None,
        verbose: bool = True
    ):
        """
        Initialize the DataTransformer for Databricks.
        
        Args:
            raw_data_path: ABFSS path to raw data file (e.g., abfss://container@storage.dfs.core.windows.net/path/file.csv)
            historical_data_path: ABFSS path to historical data file
            current_year: Current year (e.g., 2026). If None, will be extracted from path.
            current_month: Current month (1-12). If None, will be extracted from path.
            verbose: If True, print detailed progress messages. Default True.
        """
        self.raw_data_path = raw_data_path
        self.historical_data_path = historical_data_path
        
        # Store year/month or extract from path
        self._current_year = current_year
        self._current_month = current_month
        
        # Extract year and month from raw_data_path if not provided
        if raw_data_path and (current_year is None or current_month is None):
            self._extract_year_month_from_path(raw_data_path)
        
        # Cache for loaded data
        self._historical_data: Optional[pd.DataFrame] = None
        self._raw_data: Optional[pd.DataFrame] = None
        self._refined_data: Optional[pd.DataFrame] = None
        
        # Cache for forecast column mapping (month_num -> column_name)
        self._forecast_col_cache: Optional[dict] = None
        
        # Cache for historical data index (year, month) -> row for O(1) lookups
        self._historical_index: Optional[dict] = None
        
        # Verbose output control
        self.verbose = verbose
        
        # Get Spark session (available in Databricks)
        self._spark = None
        if DATABRICKS_ENV:
            try:
                self._spark = SparkSession.builder.getOrCreate()
            except Exception:
                pass
    
    # -------------------------------------------------------------------------
    # Helper: Extract year and month from ABFSS path
    # -------------------------------------------------------------------------
    def _extract_year_month_from_path(self, path: str):
        """
        Extract year and month from ABFSS path.
        
        Expected path format: .../monthly_data/2026/Feb/filename.csv
        """
        try:
            # Split path and find year/month
            parts = path.replace('\\', '/').split('/')
            
            for i, part in enumerate(parts):
                # Look for a 4-digit year
                if part.isdigit() and len(part) == 4:
                    year = int(part)
                    if 2000 <= year <= 2100:
                        if self._current_year is None:
                            self._current_year = year
                        
                        # Check if next part is a month
                        if i + 1 < len(parts):
                            month_part = parts[i + 1].lower().strip()
                            if month_part in self.MONTH_MAP:
                                if self._current_month is None:
                                    self._current_month = self.MONTH_MAP[month_part]
                        break
        except Exception as e:
            self._log(f"Warning: Could not extract year/month from path: {e}")
    
    # -------------------------------------------------------------------------
    # Logging Helper: Print message only if verbose mode is enabled
    # -------------------------------------------------------------------------
    def _log(self, message: str, force: bool = False):
        """Print message if verbose mode is enabled or force is True."""
        if self.verbose or force:
            print(message)
    
    # -------------------------------------------------------------------------
    # Cache Builder: Map month numbers (1-12) to forecast column names
    # -------------------------------------------------------------------------
    def _build_forecast_column_map(self, df: pd.DataFrame) -> dict:
        """
        Build a mapping of month numbers to forecast column names.
        
        This is cached to avoid repeated column searches.
        
        Args:
            df: DataFrame with forecast columns.
            
        Returns:
            Dictionary mapping month number (1-12) to column name.
        """
        if self._forecast_col_cache is not None:
            return self._forecast_col_cache
        
        # Pre-filter forecast columns and convert to lowercase once
        forecast_cols = [(col, col.lower()) for col in df.columns if col.lower().startswith('forecast')]
        col_map = {}
        
        for month_num in range(1, 13):
            month_abbrev = self.MONTH_ABBREV_LOWER[month_num]  # Use pre-computed lowercase
            for col, col_lower in forecast_cols:
                if f'-{month_abbrev}' in col_lower or f'_{month_abbrev}' in col_lower:
                    col_map[month_num] = col
                    break
        
        self._forecast_col_cache = col_map
        return col_map
    
    # -------------------------------------------------------------------------
    # Index Builder: Create (year, month) -> row index mapping for O(1) lookups
    # -------------------------------------------------------------------------
    def _build_historical_index(
        self,
        historical_df: pd.DataFrame,
        year_col: str = "year",
        month_col: str = "month"
    ) -> dict:
        """
        Build an index for O(1) historical data lookups by (year, month).
        
        Args:
            historical_df: Historical DataFrame.
            year_col: Column name for year.
            month_col: Column name for month.
            
        Returns:
            Dictionary mapping (year, month) to row index.
        """
        if self._historical_index is not None:
            return self._historical_index
        
        # Build index: (year, month) -> DataFrame index
        self._historical_index = {}
        for idx, row in historical_df.iterrows():
            key = (int(row[year_col]), int(row[month_col]))
            if key not in self._historical_index:  # Keep first occurrence
                self._historical_index[key] = idx
        
        return self._historical_index
    
    # -------------------------------------------------------------------------
    # Data Lookup: Get a single row from historical data by year and month
    # -------------------------------------------------------------------------
    def _get_historical_row(
        self,
        historical_df: pd.DataFrame,
        year: int,
        month: int,
        year_col: str = "year",
        month_col: str = "month"
    ) -> Optional[pd.Series]:
        """
        Get a single row from historical data for a specific year/month.
        
        Uses cached index for O(1) lookups instead of O(n) filtering.
        
        Args:
            historical_df: Historical DataFrame.
            year: Year to look up.
            month: Month to look up (1-12).
            year_col: Column name for year.
            month_col: Column name for month.
            
        Returns:
            Series representing the row, or None if not found.
        """
        # Build index if not cached
        index = self._build_historical_index(historical_df, year_col, month_col)
        
        key = (year, month)
        if key not in index:
            return None
        
        return historical_df.loc[index[key]]
    
    # -------------------------------------------------------------------------
    # Databricks Helper: Read file from ABFSS path using Spark
    # -------------------------------------------------------------------------
    def _read_from_abfss(self, path: str) -> pd.DataFrame:
        """
        Read a file from ABFSS path using Spark and convert to Pandas.
        
        Args:
            path: ABFSS path to the file (csv, parquet, or excel)
            
        Returns:
            Pandas DataFrame containing the file data.
        """
        if self._spark is None:
            raise RuntimeError("Spark session not available. This code must run in Databricks.")
        
        self._log(f"Reading from ABFSS path: {path}")
        
        path_lower = path.lower()
        
        if path_lower.endswith('.csv'):
            spark_df = self._spark.read.option("header", "true").option("inferSchema", "true").csv(path)
        elif path_lower.endswith('.parquet'):
            spark_df = self._spark.read.parquet(path)
        elif path_lower.endswith(('.xlsx', '.xls')):
            # For Excel files, use pandas directly with dbutils if available
            # Or use com.crealytics.spark.excel package if installed
            try:
                # Try using spark-excel package
                spark_df = self._spark.read.format("com.crealytics.spark.excel") \
                    .option("header", "true") \
                    .option("inferSchema", "true") \
                    .load(path)
            except Exception:
                raise ValueError(
                    f"Excel files require the spark-excel package. "
                    f"Please convert to CSV or install com.crealytics:spark-excel."
                )
        else:
            raise ValueError(f"Unsupported file format: {path}")
        
        # Convert Spark DataFrame to Pandas
        pdf = spark_df.toPandas()
        self._log(f"Loaded {len(pdf)} rows, {len(pdf.columns)} columns.")
        
        return pdf
    
    # -------------------------------------------------------------------------
    # Data Fetcher: Load historical transformed data from ABFSS path
    # -------------------------------------------------------------------------
    def fetch_historical_transformed_data(
        self, 
        path: str = None,
        force_reload: bool = False
    ) -> pd.DataFrame:
        """
        Fetch historical transformed data from ABFSS path.
        
        Args:
            path: ABFSS path to historical data file. Uses self.historical_data_path if not provided.
            force_reload: If True, reload from storage even if cached.
            
        Returns:
            DataFrame containing historical transformed data.
            
        Raises:
            ValueError: If no path is provided and historical_data_path is not set.
        """
        if self._historical_data is not None and not force_reload:
            self._log("Returning cached historical data.")
            return self._historical_data
        
        data_path = path or self.historical_data_path
        
        if not data_path:
            raise ValueError(
                "No historical data path provided. "
                "Set historical_data_path in constructor or pass path parameter."
            )
        
        self._log(f"Loading historical data from: {data_path}")
        self._historical_data = self._spark.read.table(data_path).toPandas()
        
        return self._historical_data
    
    # -------------------------------------------------------------------------
    # Data Fetcher: Load raw data from ABFSS path
    # -------------------------------------------------------------------------
    def fetch_newly_arrived_raw_data(
        self, 
        path: str = None,
        force_reload: bool = False
    ) -> pd.DataFrame:
        """
        Fetch newly arrived raw data from ABFSS path.
        
        Args:
            path: ABFSS path to raw data file. Uses self.raw_data_path if not provided.
            force_reload: If True, reload from storage even if cached.
            
        Returns:
            DataFrame containing the newly arrived raw data.
            
        Raises:
            ValueError: If no path is provided and raw_data_path is not set.
        """
        if self._raw_data is not None and not force_reload:
            self._log("Returning cached raw data.")
            return self._raw_data
        
        data_path = path or self.raw_data_path
        
        if not data_path:
            raise ValueError(
                "No raw data path provided. "
                "Set raw_data_path in constructor or pass path parameter."
            )
        
        self._log(f"Loading raw data from: {data_path}")
        self._raw_data = self._read_from_abfss(data_path)
        
        return self._raw_data
    
    # -------------------------------------------------------------------------
    # Data Filter: Remove rows with specified Revenue_Category values
    # -------------------------------------------------------------------------
    def filter_revenue_categories(
        self, 
        df: pd.DataFrame,
        column: str = "Revenue_Category",
        exclude_values: list = None
    ) -> pd.DataFrame:
        """
        Filter out rows with specified Revenue_Category values.
        
        By default, removes rows where Revenue_Category is 'Risk', 'Opportunity', or 'Un-id'.
        
        Args:
            df: DataFrame to filter.
            column: Column name to filter on. Defaults to 'Revenue_Category'.
            exclude_values: List of values to exclude. Defaults to ['Risk', 'Opportunity', 'Un-id'].
            
        Returns:
            Filtered DataFrame with excluded rows removed.
            
        Raises:
            ValueError: If the specified column does not exist in the DataFrame.
        """
        if exclude_values is None:
            exclude_values = ['Risk', 'Opportunity', 'Un-id']
        
        if column not in df.columns:
            raise ValueError(
                f"Column '{column}' not found in DataFrame.\n"
                f"Available columns: {list(df.columns)}"
            )
        
        original_count = len(df)
        
        # Filter out rows where column value is in exclude_values (returns new DataFrame)
        filtered_df = df.loc[~df[column].isin(exclude_values)]
        
        removed_count = original_count - len(filtered_df)
        
        self._log(f"Filtered Revenue_Category: removed {removed_count} rows "
              f"(values: {exclude_values})")
        self._log(f"Rows remaining: {len(filtered_df)} (was {original_count})")
        
        return filtered_df
    
    # -------------------------------------------------------------------------
    # Data Normalizer: Convert decimal probabilities (0.5) to percentages (50)
    # -------------------------------------------------------------------------
    def normalize_probability_pct(
        self,
        df: pd.DataFrame,
        column: str = "ProbabilityPer"
    ) -> pd.DataFrame:
        """
        Normalize probability percentage values.
        
        If value > 1, keep as is; else multiply by 100.
        This handles cases where some values are in decimal form (0.5 = 50%)
        and others are already percentages (50 = 50%).
        
        Args:
            df: DataFrame to process.
            column: Column name containing probability values.
            
        Returns:
            DataFrame with normalized probability values.
        """
        result = df.copy()
        
        if column not in result.columns:
            self._log(f"Warning: Column '{column}' not found. Skipping probability normalization.")
            return result
        
        # Vectorized operation (faster than apply)
        mask = result[column] <= 1
        decimal_count = mask.sum()
        result.loc[mask, column] = result.loc[mask, column] * 100
        
        self._log(f"Normalized {column}: converted {decimal_count} decimal values to percentages.")
        
        return result
    
    # -------------------------------------------------------------------------
    # Data Cleaner: Remove rows where category='-' and all forecasts are zero
    # -------------------------------------------------------------------------
    def remove_empty_forecast_rows(
        self,
        df: pd.DataFrame,
        category_column: str = "Revenue_Category",
        category_value: str = "-",
        forecast_pattern: str = "Forecast"
    ) -> pd.DataFrame:
        """
        Remove rows where Revenue_Category equals a specific value AND all forecast columns are zero.
        
        Args:
            df: DataFrame to process.
            category_column: Column name for revenue category.
            category_value: Category value to check (default: '-').
            forecast_pattern: Pattern to identify forecast columns (uses filter with 'like').
            
        Returns:
            DataFrame with empty forecast rows removed.
        """
        result = df.copy()
        
        if category_column not in result.columns:
            self._log(f"Warning: Column '{category_column}' not found. Skipping empty forecast removal.")
            return result
        
        # Find forecast columns
        forecast_cols = result.filter(like=forecast_pattern).columns
        
        if len(forecast_cols) == 0:
            self._log(f"Warning: No columns matching '{forecast_pattern}' found. Skipping empty forecast removal.")
            return result
        
        original_count = len(result)
        
        # Create mask: category matches AND all forecast columns are zero
        mask = (result[category_column] == category_value) & (result[forecast_cols].eq(0).all(axis=1))
        result = result.loc[~mask]  # No copy needed - loc returns view, we already copied at start
        
        removed_count = original_count - len(result)
        self._log(f"Removed {removed_count} rows where {category_column}='{category_value}' and all forecasts are zero.")
        
        return result
    
    # -------------------------------------------------------------------------
    # Data Transformer: Replace a specific value in Revenue_Category column
    # -------------------------------------------------------------------------
    def replace_category_value(
        self,
        df: pd.DataFrame,
        column: str = "Revenue_Category",
        old_value: str = "-",
        new_value: str = "Actuals"
    ) -> pd.DataFrame:
        """
        Replace a specific value in the Revenue_Category column.
        
        Args:
            df: DataFrame to process.
            column: Column name to modify.
            old_value: Value to replace.
            new_value: Replacement value.
            
        Returns:
            DataFrame with replaced values.
        """
        result = df.copy()
        
        if column not in result.columns:
            self._log(f"Warning: Column '{column}' not found. Skipping value replacement.")
            return result
        
        replace_count = (result[column] == old_value).sum()
        result[column] = result[column].replace(old_value, new_value)
        
        self._log(f"Replaced '{old_value}' with '{new_value}' in {column}: {replace_count} rows updated.")
        
        return result
    
    # -------------------------------------------------------------------------
    # Data Pipeline: Apply all refinement steps (filter, normalize, clean)
    # -------------------------------------------------------------------------
    def refine_raw_data(
        self,
        df: pd.DataFrame = None,
        Revenue_Category_col: str = "Revenue_Category",
        ProbabilityPer_col: str = "ProbabilityPer",
        force_reload: bool = False
    ) -> pd.DataFrame:
        """
        Apply all refinement steps to raw data in sequence.
        
        Steps:
        1. Filter out 'Opportunity', 'Risk', 'Un-id' from Revenue_Category
        2. Normalize probability percentage (if <= 1, multiply by 100)
        3. Remove rows where Revenue_Category == '-' AND all forecast columns are zero
        4. Replace '-' with 'Actuals' in Revenue_Category
        
        Args:
            df: Raw DataFrame to refine. If None, uses cached raw data.
            Revenue_Category_col: Column name for revenue category.
            ProbabilityPer_col: Column name for probability percentage.
            force_reload: If True, ignore cached refined data.
            
        Returns:
            Refined DataFrame.
        """
        # Return cached if available
        if self._refined_data is not None and not force_reload and df is None:
            self._log("Returning cached refined data.")
            return self._refined_data
        
        # Use cached raw data if df not provided
        if df is None:
            if self._raw_data is None:
                raise ValueError("No raw data available. Call fetch_newly_arrived_raw_data() first or provide df.")
            df = self._raw_data
        
        self._log("\n" + "="*60)
        self._log("Starting raw data refinement...")
        self._log("="*60)
        
        result = df.copy()
        original_count = len(result)
        
        # Step 1: Filter out unwanted revenue categories
        result = self.filter_revenue_categories(
            result, 
            column=Revenue_Category_col,
            exclude_values=['Opportunity', 'Risk', 'Un-id']
        )
        
        # Step 2: Normalize probability percentage
        result = self.normalize_probability_pct(result, column=ProbabilityPer_col)
        
        # Step 3: Remove rows where category is '-' and all forecasts are zero
        result = self.remove_empty_forecast_rows(
            result,
            category_column=Revenue_Category_col,
            category_value="-"
        )
        
        # Step 4: Replace '-' with 'Actuals'
        result = self.replace_category_value(
            result,
            column=Revenue_Category_col,
            old_value="-",
            new_value="Actuals"
        )
        
        self._log("="*60)
        self._log(f"Refinement complete: {original_count} → {len(result)} rows")
        self._log("="*60 + "\n")
        
        # Cache the refined data
        self._refined_data = result
        
        return result
    
    # -------------------------------------------------------------------------
    # Feature Creator: Generate lag features (LM1-3, YTD, LY1-2, deltas, etc.)
    # -------------------------------------------------------------------------
    def create_lag_features(
        self,
        df: pd.DataFrame,
        current_month: int,
        current_year: int = None,
        historical_df: pd.DataFrame = None,
        revenue_category_col: str = "Revenue_Category",
        actuals_value: str = "Actuals",
        year_col: str = "year",
        month_col: str = "month"
    ) -> dict:
        """
        Create lag features from raw data and historical data.
        
        IMPORTANT: For early months (Jan, Feb, Mar), LM1/LM2/LM3 may reference
        months from the previous year. In these cases, the function automatically
        uses historical data's revenue column instead of raw data's forecast columns.
        
        Example edge cases:
        - Jan 2025: LM1=Dec 2024, LM2=Nov 2024, LM3=Oct 2024 (all from historical)
        - Feb 2025: LM1=Jan 2025 (raw), LM2=Dec 2024 (historical), LM3=Nov 2024 (historical)
        - Mar 2025: LM1=Feb 2025 (raw), LM2=Jan 2025 (raw), LM3=Dec 2024 (historical)
        
        From Raw Data (or Historical if month is from previous year):
        - LM1: Last month's revenue (Forecast column or historical revenue)
        - LM2: 2 months ago revenue (Forecast column or historical revenue)
        - LM3: 3 months ago revenue (Forecast column or historical revenue)
        - Avg_LM1_LM2_LM3: Average of LM1, LM2, LM3
        - YTD: Sum of all Forecast columns from Jan up to (excluding) current month
        
        From Historical Data (if provided):
        - LY1: Sum of revenue for all rows where year=current_year-1 (full year)
        - LY2: Sum of revenue for all rows where year=current_year-2 (full year)
        - Avg_LY1_LY2: Average of LY1 and LY2
        - LY1CM: Last year same month's revenue
        - LY2CM: 2 years ago same month's revenue
        - LY1LM1: Last Year's LM1 (from row where year=current_year-1, month=current_month)
        - LY1LM2: Last Year's LM2
        - LY1LM3: Last Year's LM3
        - LY1_YTD: Last Year's YTD (same month)
        - LY2_YTD: 2 Years Ago YTD (same month)
        - Avg_LM1_LY1LM1: Average of current LM1 and LY1LM1
        - LY1NM1: Last Year's Next Month's revenue (e.g., Jan 2026 -> Feb 2025's revenue)
        - Delta_LM1_LM2: LM1 - LM2
        - Delta_YTD_LY1: YTD - LY1
        - Delta_LY1_LY2: LY1 - LY2
        - Delta_YTD_LY2: YTD - LY2
        - ROLL3M_STD: Rolling std dev of last 3 months' revenue (excluding current)
        - ROLL6M_STD: Rolling std dev of last 6 months' revenue (excluding current)
        - YoY_growth_rate: (LM1 - LY1LM1) / LY1LM1
        - YoY_YTD_growth: (YTD - LY1_YTD) / LY1_YTD
        - LY1_CM_trend_slope_3: Slope of revenue from months 14, 13, 12 ago (shift=12, rolling=3)
        - MOM_LM1_LM3: LM1 - LM3
        - LY1_ROLL3M_SLOPE: Slope of revenue from months 15, 14, 13 ago (shift=13, rolling=3)
        
        Example: If current is 2025 Nov:
        - LM1 uses Forecast-Oct-* column (from raw data)
        - LY1: Sum of revenue for all 2024 rows
        - LY2: Sum of revenue for all 2023 rows
        - LY1CM: revenue from row where year=2024, month=11
        - LY2CM: revenue from row where year=2023, month=11
        - LY1LM1/LY1LM2/LY1LM3/LY1_YTD: Get from row where year=2024, month=11
        - LY2_YTD: Get YTD from row where year=2023, month=11
        - LY1NM1: Get LM1 from row where year=2024, month=12 (December)
        
        Args:
            df: DataFrame with refined raw data (must have Actuals in revenue_category).
            current_month: Current month as integer (1-12).
            current_year: Current year (e.g., 2025). Required for historical features.
            historical_df: DataFrame with historical transformed data. Required for historical features.
            revenue_category_col: Column name for revenue category.
            actuals_value: Value representing actuals in revenue_category.
            year_col: Column name for year in historical data.
            month_col: Column name for month in historical data.
            
        Returns:
            Dictionary with all lag features.
        """
        # DEBUG: Show input parameters
        self._log(f"\n{'='*60}")
        self._log(f"CREATE_LAG_FEATURES INPUT PARAMETERS:")
        self._log(f"  current_month: {current_month}")
        self._log(f"  current_year: {current_year}")
        self._log(f"{'='*60}")
        
        if current_month < 1 or current_month > 12:
            raise ValueError(f"current_month must be 1-12, got {current_month}")
        
        if current_year is None:
            raise ValueError("current_year is required for lag feature calculation.")
        
        # Force rebuild historical index to ensure fresh data lookups
        self._historical_index = None
        
        if revenue_category_col not in df.columns:
            raise ValueError(f"Column '{revenue_category_col}' not found in DataFrame.")
        
        # Filter to only Actuals rows
        actuals_df = df[df[revenue_category_col] == actuals_value]
        
        # Flag to track if we have actuals data from raw data
        has_actuals_in_raw = len(actuals_df) > 0
        
        if not has_actuals_in_raw:
            self._log(f"Warning: No rows with {revenue_category_col}='{actuals_value}' found in raw data.")
            self._log(f"  Will use historical data for lag features if available.")
        
        # Calculate lag months with year wrap-around
        def get_prev_month_year(year: int, month: int, offset: int) -> tuple:
            """
            Get (year, month) tuple with offset going backwards.
            Handles year boundary wrap-around.
            
            Example: (2025, 1) with offset=1 returns (2024, 12)
            """
            result_month = month - offset
            result_year = year
            while result_month <= 0:
                result_month += 12
                result_year -= 1
            return (result_year, result_month)
        
        # Get year and month for each lag period
        lm1_year, lm1_month = get_prev_month_year(current_year, current_month, 1)
        lm2_year, lm2_month = get_prev_month_year(current_year, current_month, 2)
        lm3_year, lm3_month = get_prev_month_year(current_year, current_month, 3)
        
        self._log(f"\n{'='*60}")
        self._log(f"Creating lag features for: {current_year or ''} {self.MONTH_ABBREV[current_month]}")
        self._log(f"{'='*60}")
        
        self._log(f"\n--- Lag Month Calculations ---")
        self._log(f"  LM1: {lm1_year} {self.MONTH_ABBREV[lm1_month]} (month {lm1_month})")
        self._log(f"  LM2: {lm2_year} {self.MONTH_ABBREV[lm2_month]} (month {lm2_month})")
        self._log(f"  LM3: {lm3_year} {self.MONTH_ABBREV[lm3_month]} (month {lm3_month})")
        
        # Helper function to get LM value from appropriate source
        def get_lm_value(lm_year: int, lm_month: int, label: str) -> float:
            """
            Get lag month value from the appropriate data source.
            
            - If lm_year < current_year (previous year): use historical data's revenue column
            - If lm_year == current_year: use raw data's forecast column (if actuals exist)
            - If raw data doesn't have actuals or the column, fall back to historical data
            """
            # Check if we need to use historical data
            use_historical = (lm_year < current_year) or (not has_actuals_in_raw)
            
            if use_historical:
                # Use historical data's revenue column
                reason = "previous year" if lm_year < current_year else "no actuals in raw data"
                self._log(f"\n  {label}: Using historical data ({reason})")
                
                if historical_df is None:
                    self._log(f"    Warning: No historical data provided. {label} set to 0.")
                    return 0.0
                
                hist_row = self._get_historical_row(historical_df, lm_year, lm_month, year_col, month_col)
                if hist_row is None:
                    self._log(f"    Warning: No historical data found for {lm_year} {self.MONTH_ABBREV[lm_month]}")
                    return 0.0
                
                if 'revenue' not in historical_df.columns:
                    self._log(f"    Warning: 'revenue' column not found in historical data")
                    return 0.0
                
                value = float(hist_row['revenue'])
                self._log(f"    {label}: {lm_year} {self.MONTH_ABBREV[lm_month]} revenue = {value:,.2f}")
                return value
            else:
                # Current year with actuals available - use raw data's forecast column
                self._log(f"\n  {label}: Using raw data (current year)")
                value = self._sum_forecast_column(actuals_df, lm_month, label)
                
                # If raw data doesn't have this month, fall back to historical
                if value == 0.0 and historical_df is not None:
                    self._log(f"    Checking historical data as fallback...")
                    hist_row = self._get_historical_row(historical_df, lm_year, lm_month, year_col, month_col)
                    if hist_row is not None and 'revenue' in historical_df.columns:
                        fallback_value = float(hist_row['revenue'])
                        if fallback_value != 0.0:
                            self._log(f"    Using historical fallback: {fallback_value:,.2f}")
                            return fallback_value
                
                return value
        
        # Get LM1, LM2, LM3 values from appropriate sources
        lm1 = get_lm_value(lm1_year, lm1_month, "LM1")
        lm2 = get_lm_value(lm2_year, lm2_month, "LM2")
        lm3 = get_lm_value(lm3_year, lm3_month, "LM3")
        
        # Calculate average
        avg_lm1_lm2_lm3 = (lm1 + lm2 + lm3) / 3
        
        self._log(f"\n--- Lag Feature Summary ---")
        self._log(f"  LM1: {lm1:,.2f}")
        self._log(f"  LM2: {lm2:,.2f}")
        self._log(f"  LM3: {lm3:,.2f}")
        self._log(f"  Avg_LM1_LM2_LM3: {avg_lm1_lm2_lm3:,.2f}")
        
        # Calculate YTD: sum of all forecast columns from Jan to (current_month - 1)
        # Note: YTD only considers current year months in raw data
        if has_actuals_in_raw:
            ytd = self._calculate_ytd(actuals_df, current_month)
        else:
            # No actuals in raw data - YTD from raw is 0
            # For January, YTD is always 0 anyway
            ytd = 0.0
            self._log(f"  YTD: 0 (no actuals in raw data, or current month is January)")
        
        result = {
            'LM1': lm1,
            'LM2': lm2,
            'LM3': lm3,
            'Avg_LM1_LM2_LM3': avg_lm1_lm2_lm3,
            'YTD': ytd
        }
        
        # --- Historical Data Features ---
        # Diagnostic: Check if historical data is available
        self._log(f"\n--- Historical Data Check ---")
        self._log(f"  historical_df provided: {historical_df is not None}")
        self._log(f"  current_year: {current_year}")
        
        if historical_df is not None and current_year is not None:
            # Diagnostic: Show available year/month range in historical data
            self._log(f"  historical_df shape: {historical_df.shape}")
            self._log(f"  historical_df columns: {list(historical_df.columns)[:10]}...")  # First 10 columns
            
            if year_col in historical_df.columns and month_col in historical_df.columns:
                min_year = int(historical_df[year_col].min())
                max_year = int(historical_df[year_col].max())
                latest_row = historical_df.loc[historical_df[year_col] == max_year]
                max_month = int(latest_row[month_col].max()) if len(latest_row) > 0 else 0
                self._log(f"\n--- Historical Data Range ---")
                self._log(f"  Year range: {min_year} - {max_year}")
                self._log(f"  Latest month in {max_year}: {self.MONTH_ABBREV.get(max_month, max_month)}")
                self._log(f"  Required for lag features: {lm3_year} {self.MONTH_ABBREV[lm3_month]} to {lm1_year} {self.MONTH_ABBREV[lm1_month]}")
            else:
                self._log(f"  WARNING: '{year_col}' or '{month_col}' column not found in historical data!")
                self._log(f"  Available columns: {list(historical_df.columns)}")
            self._log(f"\n--- Historical Data Features ---")
            
            # Validate required columns in historical data
            required_cols = [year_col, month_col, 'LM1', 'LM2', 'LM3', 'YTD', 'revenue']
            missing_cols = [col for col in required_cols if col not in historical_df.columns]
            
            if missing_cols:
                self._log(f"  Warning: Missing columns in historical data: {missing_cols}")
                self._log(f"  Skipping historical features.")
            else:
                last_year = current_year - 1
                two_years_ago = current_year - 2
                self._log(f"  Looking up last year: {last_year} {self.MONTH_ABBREV[current_month]}")
                
                # --- LY1 & LY2: Full year revenue sums (optimized with boolean indexing) ---
                ly1_mask = historical_df[year_col] == last_year
                ly2_mask = historical_df[year_col] == two_years_ago
                
                ly1 = float(historical_df.loc[ly1_mask, 'revenue'].sum())
                ly2 = float(historical_df.loc[ly2_mask, 'revenue'].sum())
                avg_ly1_ly2 = (ly1 + ly2) / 2 if (ly1 + ly2) != 0 else 0.0
                
                self._log(f"  LY1 (full year {last_year}): {ly1:,.2f}")
                self._log(f"  LY2 (full year {two_years_ago}): {ly2:,.2f}")
                self._log(f"  Avg_LY1_LY2: {avg_ly1_ly2:,.2f}")
                
                # Use helper method for row lookups (optimized)
                ly1_row = self._get_historical_row(historical_df, last_year, current_month, year_col, month_col)
                
                if ly1_row is None:
                    self._log(f"  Warning: No data found for {last_year} month {current_month}")
                    ly1lm1, ly1lm2, ly1lm3, ly1_ytd, ly1cm = 0.0, 0.0, 0.0, 0.0, 0.0
                else:
                    ly1lm1 = float(ly1_row['LM1'])
                    ly1lm2 = float(ly1_row['LM2'])
                    ly1lm3 = float(ly1_row['LM3'])
                    ly1_ytd = float(ly1_row['YTD'])
                    ly1cm = float(ly1_row['revenue'])
                
                self._log(f"  LY1LM1: {ly1lm1:,.2f}")
                self._log(f"  LY1LM2: {ly1lm2:,.2f}")
                self._log(f"  LY1LM3: {ly1lm3:,.2f}")
                self._log(f"  LY1_YTD: {ly1_ytd:,.2f}")
                self._log(f"  LY1CM: {ly1cm:,.2f}")
                
                # Use helper method for 2 years ago lookup
                self._log(f"  Looking up 2 years ago: {two_years_ago} {self.MONTH_ABBREV[current_month]}")
                ly2_row = self._get_historical_row(historical_df, two_years_ago, current_month, year_col, month_col)
                
                if ly2_row is None:
                    self._log(f"  Warning: No data found for {two_years_ago} month {current_month}")
                    ly2_ytd, ly2cm = 0.0, 0.0
                else:
                    ly2_ytd = float(ly2_row['YTD'])
                    ly2cm = float(ly2_row['revenue'])
                
                self._log(f"  LY2_YTD: {ly2_ytd:,.2f}")
                self._log(f"  LY2CM: {ly2cm:,.2f}")
                
                # Calculate Avg_LM1_LY1LM1
                avg_lm1_ly1lm1 = (lm1 + ly1lm1) / 2
                self._log(f"  Avg_LM1_LY1LM1: {avg_lm1_ly1lm1:,.2f}")
                
                # Find LY1NM1: Last Year's Next Month's Revenue
                # For Jan 2026: next month = Feb, so look up Feb 2025's revenue
                # For Dec 2025: next month = Jan, so look up Jan 2025 (wraps to current year)
                # For Jan 2026: next month = Feb, so look up Feb 2025 (last year)
                nm_month = current_month + 1
                if nm_month > 12:
                    nm_month = 1
                    nm_year = current_year  # Next month wraps to current year's January
                else:
                    nm_year = last_year  # Stay in last year
                
                self._log(f"  LY1NM1 calculation:")
                self._log(f"    current_month: {current_month}, current_year: {current_year}, last_year: {last_year}")
                self._log(f"    nm_month (current_month+1): {current_month + 1} -> {nm_month}")
                self._log(f"    nm_year: {nm_year}")
                self._log(f"  Looking up LY1NM1: {nm_year} {self.MONTH_ABBREV[nm_month]}")
                
                # DEBUG: Show what (year, month) entries exist in historical data around 2025
                if self._historical_index:
                    nearby_keys = [k for k in self._historical_index.keys() if k[0] in (2024, 2025)]
                    self._log(f"  DEBUG: Historical index entries for 2024-2025: {sorted(nearby_keys)}")
                nm_row = self._get_historical_row(historical_df, nm_year, nm_month, year_col, month_col)
                
                if nm_row is None:
                    self._log(f"  Warning: No data found for {nm_year} month {nm_month}")
                    ly1nm1 = 0.0
                else:
                    # DEBUG: Show what row was actually returned
                    actual_year = nm_row[year_col] if year_col in nm_row.index else 'N/A'
                    actual_month = nm_row[month_col] if month_col in nm_row.index else 'N/A'
                    self._log(f"  DEBUG: Row returned has year={actual_year}, month={actual_month}")
                    # Use 'revenue' column (actual revenue of that month), not 'LM1' (which would be previous month's revenue)
                    ly1nm1 = float(nm_row['revenue'])
                
                self._log(f"  LY1NM1: {ly1nm1:,.2f}")
                
                # --- Delta features ---
                delta_lm1_lm2 = lm1 - lm2
                delta_ytd_ly1 = ytd - ly1
                delta_ly1_ly2 = ly1 - ly2
                delta_ytd_ly2 = ytd - ly2
                
                self._log(f"\n--- Delta Features ---")
                self._log(f"  Delta_LM1_LM2: {delta_lm1_lm2:,.2f}")
                self._log(f"  Delta_YTD_LY1: {delta_ytd_ly1:,.2f}")
                self._log(f"  Delta_LY1_LY2: {delta_ly1_ly2:,.2f}")
                self._log(f"  Delta_YTD_LY2: {delta_ytd_ly2:,.2f}")
                
                # --- Rolling Standard Deviation Features (optimized with helper) ---
                def get_past_months(cur_year: int, cur_month: int, n_months: int) -> list:
                    """Get list of (year, month) tuples for n months before current."""
                    months = []
                    y, m = cur_year, cur_month
                    for _ in range(n_months):
                        m -= 1
                        if m == 0:
                            m = 12
                            y -= 1
                        months.append((y, m))
                    return months
                
                # Get revenue values using helper method (optimized)
                past_3_months = get_past_months(current_year, current_month, 3)
                past_6_months = get_past_months(current_year, current_month, 6)
                
                self._log(f"\n--- Rolling Std Features ---")
                self._log(f"  ROLL3M months: {[(y, self.MONTH_ABBREV[m]) for y, m in past_3_months]}")
                self._log(f"  ROLL6M months: {[(y, self.MONTH_ABBREV[m]) for y, m in past_6_months]}")
                
                # Use helper method for lookups (reduces repeated filtering)
                revenues_3m = [float(r['revenue']) for y, m in past_3_months 
                              if (r := self._get_historical_row(historical_df, y, m, year_col, month_col)) is not None]
                revenues_6m = [float(r['revenue']) for y, m in past_6_months 
                              if (r := self._get_historical_row(historical_df, y, m, year_col, month_col)) is not None]
                
                # Calculate standard deviations (need at least 2 values)
                roll3m_std = float(np.std(revenues_3m, ddof=1)) if len(revenues_3m) >= 2 else 0.0
                roll6m_std = float(np.std(revenues_6m, ddof=1)) if len(revenues_6m) >= 2 else 0.0
                
                self._log(f"  ROLL3M values: {[f'{v:,.0f}' for v in revenues_3m]} ({len(revenues_3m)} found)")
                self._log(f"  ROLL6M values: {[f'{v:,.0f}' for v in revenues_6m]} ({len(revenues_6m)} found)")
                self._log(f"  ROLL3M_STD: {roll3m_std:,.2f}")
                self._log(f"  ROLL6M_STD: {roll6m_std:,.2f}")
                
                # --- Growth & Trend Features ---
                self._log(f"\n--- Growth & Trend Features ---")
                
                yoy_growth_rate = (lm1 - ly1lm1) / ly1lm1 if ly1lm1 != 0 else 0.0
                self._log(f"  YoY_growth_rate: {yoy_growth_rate:.4f} ({yoy_growth_rate*100:.2f}%)")
                
                yoy_ytd_growth = (ytd - ly1_ytd) / ly1_ytd if ly1_ytd != 0 else 0.0
                self._log(f"  YoY_YTD_growth: {yoy_ytd_growth:.4f} ({yoy_ytd_growth*100:.2f}%)")
                
                mom_lm1_lm3 = lm1 - lm3
                self._log(f"  MOM_LM1_LM3: {mom_lm1_lm3:,.2f}")
                
                # --- LY1_CM_trend_slope_3: df["revenue"].shift(12).rolling(3) ---
                # Uses months 12, 13, 14 ago (ordered oldest to newest for slope)
                # For Jan 2026: Nov 2024, Dec 2024, Jan 2025
                self._log(f"\n--- LY1_CM_trend_slope_3 (shift=12, rolling=3) ---")
                
                # get_past_months returns list where [0]=1 month ago, [n-1]=n months ago
                past_14 = get_past_months(current_year, current_month, 14)
                ly1_cm_months = [
                    past_14[13],  # 14 months ago (oldest)
                    past_14[12],  # 13 months ago (middle)
                    past_14[11],  # 12 months ago (newest)
                ]
                
                self._log(f"  LY1_CM months: {[(y, self.MONTH_ABBREV[m]) for y, m in ly1_cm_months]}")
                
                ly1_cm_revenues = []
                for y, m in ly1_cm_months:
                    row = self._get_historical_row(historical_df, y, m, year_col, month_col)
                    if row is not None and 'revenue' in historical_df.columns:
                        ly1_cm_revenues.append(float(row['revenue']))
                    else:
                        ly1_cm_revenues.append(0.0)
                
                self._log(f"  LY1_CM values: {[f'{v:,.0f}' for v in ly1_cm_revenues]}")
                
                if len(ly1_cm_revenues) >= 3 and any(v != 0 for v in ly1_cm_revenues):
                    x_vals = np.array([1, 2, 3])
                    y_vals = np.array(ly1_cm_revenues)
                    slope, _ = np.polyfit(x_vals, y_vals, 1)
                    ly1_cm_trend_slope_3 = float(slope)
                else:
                    ly1_cm_trend_slope_3 = 0.0
                
                self._log(f"  LY1_CM_trend_slope_3: {ly1_cm_trend_slope_3:,.2f}")
                self._log(f"    (slope of {[f'{v:,.0f}' for v in ly1_cm_revenues]} vs [1,2,3])")
                
                # --- LY1_ROLL3M_SLOPE: df["revenue"].shift(13).rolling(3) ---
                # Uses months 13, 14, 15 ago (ordered oldest to newest for slope)
                # For Jan 2026: Oct 2024, Nov 2024, Dec 2024
                self._log(f"\n--- LY1_ROLL3M_SLOPE (shift=13, rolling=3) ---")
                
                # get_past_months returns list where [0]=1 month ago, [n-1]=n months ago
                past_15 = get_past_months(current_year, current_month, 15)
                ly1_roll_months = [
                    past_15[14],  # 15 months ago (oldest)
                    past_15[13],  # 14 months ago (middle)
                    past_15[12],  # 13 months ago (newest)
                ]
                
                self._log(f"  LY1_ROLL3M months: {[(y, self.MONTH_ABBREV[m]) for y, m in ly1_roll_months]}")
                
                ly1_roll_revenues = []
                for y, m in ly1_roll_months:
                    row = self._get_historical_row(historical_df, y, m, year_col, month_col)
                    if row is not None and 'revenue' in historical_df.columns:
                        ly1_roll_revenues.append(float(row['revenue']))
                    else:
                        ly1_roll_revenues.append(0.0)
                
                self._log(f"  LY1_ROLL3M values: {[f'{v:,.0f}' for v in ly1_roll_revenues]}")
                
                if len(ly1_roll_revenues) >= 3 and any(v != 0 for v in ly1_roll_revenues):
                    x_vals = np.array([1, 2, 3])
                    y_vals = np.array(ly1_roll_revenues)
                    slope, _ = np.polyfit(x_vals, y_vals, 1)
                    ly1_roll3m_slope = float(slope)
                else:
                    ly1_roll3m_slope = 0.0
                
                self._log(f"  LY1_ROLL3M_SLOPE: {ly1_roll3m_slope:,.2f}")
                self._log(f"    (slope of {[f'{v:,.0f}' for v in ly1_roll_revenues]} vs [1,2,3])")
                
                # Add historical features to result
                result['LY1'] = ly1
                result['LY2'] = ly2
                result['Avg_LY1_LY2'] = avg_ly1_ly2
                result['LY1CM'] = ly1cm
                result['LY2CM'] = ly2cm
                result['LY1LM1'] = ly1lm1
                result['LY1LM2'] = ly1lm2
                result['LY1LM3'] = ly1lm3
                result['LY1_YTD'] = ly1_ytd
                result['LY2_YTD'] = ly2_ytd
                result['Avg_LM1_LY1LM1'] = avg_lm1_ly1lm1
                result['LY1NM1'] = ly1nm1
                result['Delta_LM1_LM2'] = delta_lm1_lm2
                result['Delta_YTD_LY1'] = delta_ytd_ly1
                result['Delta_LY1_LY2'] = delta_ly1_ly2
                result['Delta_YTD_LY2'] = delta_ytd_ly2
                result['ROLL3M_STD'] = roll3m_std
                result['ROLL6M_STD'] = roll6m_std
                result['YoY_growth_rate'] = yoy_growth_rate
                result['YoY_YTD_growth'] = yoy_ytd_growth
                result['LY1_CM_trend_slope_3'] = ly1_cm_trend_slope_3
                result['MOM_LM1_LM3'] = mom_lm1_lm3
                result['LY1_ROLL3M_SLOPE'] = ly1_roll3m_slope
        
        # Print summary
        self._log(f"\n{'='*60}")
        self._log(f"All lag features calculated:")
        self._log(f"{'='*60}")
        for key, value in result.items():
            self._log(f"  {key}: {value:,.2f}")
        
        return result
    
    # ------------------------------------------------------------------------
    # Feature Creator: Generate revenue ratios and forecast-based features
    # ------------------------------------------------------------------------
    def create_revenue_based_features(
        self,
        df: pd.DataFrame,
        current_month: int,
        current_year: int = None,
        ytd: float = None,
        historical_df: pd.DataFrame = None,
        revenue_category_col: str = "Revenue_Category",
        committed_signed_value: str = "Committed - Signed",
        committed_unsigned_value: str = "Committed - Unsigned",
        wtd_pipeline_value: str = "Wtd. Pipeline",
        year_col: str = "year",
        month_col: str = "month"
    ) -> dict:
        """
        Create revenue-based ratio features from incremental raw data.
        
        Uses only forecast columns from current month until end of year (Dec).
        Example: If current_month=10 (October), uses Forecast columns for Oct, Nov, Dec.
        
        Features calculated:
        - Committed_Signed_revenue_ratio: sum(Committed-Signed forecast) / sum(all forecast)
        - Committed_Unsigned_revenue_ratio: sum(Committed-Unsigned forecast) / sum(all forecast)
        - Wtd_Pipeline_revenue_ratio: sum(Wtd. Pipeline forecast) / sum(all forecast)
        - wtd_pip_commit_ratio: Wtd_Pipeline / (Committed_Signed + Committed_Unsigned)
        - commit_sign_unsign_ratio: Committed_Signed / Committed_Unsigned
        - signed_vs_committed_ratio: Committed_Signed / (Committed_Signed + Committed_Unsigned)
        - fcst_committed_signed_rem: Forecast sum (current to Dec) for Committed - Signed rows
        - fcst_committed_unsigned_rem: Forecast sum (current to Dec) for Committed - Unsigned rows
        - fcst_wtd_pipeline_rem: Forecast sum (current to Dec) for Wtd. Pipeline rows
        - Forecast_Jan, Forecast_Feb, ..., Forecast_Dec: Sum of each month's forecast column
        - Forecast_yearly: Sum of all forecast columns from Jan to Dec (full year)
        - forecast_roy: Sum of forecast columns from current month to December (Rest of Year)
        - total_per_month_left: forecast_roy / number of months remaining (current to Dec)
        - year_end_rev: YTD + forecast_roy
        - Forecast_gap: forecast_roy - YTD
        - prev_month_achv_pct: Previous month's achievement % (handles year wrap-around for January)
        
        Args:
            df: DataFrame with raw data (should have Revenue_Category column and Forecast-* columns).
            current_month: Current month (1-12). Only uses forecast columns from this month to December.
            current_year: Current year (e.g., 2025). Required for prev_month_achv_pct.
            ytd: Year-to-date revenue (from lag features). Required for year_end_rev and Forecast_gap.
            historical_df: DataFrame with historical transformed data. Required for prev_month_achv_pct.
            revenue_category_col: Column name for revenue category.
            committed_signed_value: Value for Committed - Signed in revenue_category.
            committed_unsigned_value: Value for Committed - Unsigned in revenue_category.
            wtd_pipeline_value: Value for Wtd. Pipeline in revenue_category.
            year_col: Column name for year in historical data.
            month_col: Column name for month in historical data.
            
        Returns:
            Dictionary with revenue-based ratio features.
        """
        if current_month < 1 or current_month > 12:
            raise ValueError(f"current_month must be 1-12, got {current_month}")
        
        if revenue_category_col not in df.columns:
            raise ValueError(f"Column '{revenue_category_col}' not found in DataFrame.")
        
        # Get months from current month to December (end of year)
        months_to_include = list(range(current_month, 13))  # current_month to 12
        month_abbrevs_to_include = [self.MONTH_ABBREV[m].lower() for m in months_to_include]
        
        # Find all Forecast columns first
        all_forecast_cols = [col for col in df.columns if col.lower().startswith('forecast')]
        
        # Filter to only include columns for current month through December
        forecast_cols = []
        for col in all_forecast_cols:
            col_lower = col.lower()
            for month_abbrev in month_abbrevs_to_include:
                # Check if month abbreviation appears in column name (e.g., 'forecast-oct-...')
                if f'-{month_abbrev}' in col_lower or f'_{month_abbrev}' in col_lower:
                    forecast_cols.append(col)
                    break
        
        if not forecast_cols:
            self._log(f"Warning: No Forecast columns found for months {current_month}-12.")
            return {
                'Committed_Signed_revenue_ratio': 0.0,
                'Committed_Unsigned_revenue_ratio': 0.0,
                'Wtd_Pipeline_revenue_ratio': 0.0,
                'wtd_pip_commit_ratio': 0.0,
                'commit_sign_unsign_ratio': 0.0,
                'signed_vs_committed_ratio': 0.0
            }
        
        self._log(f"\n{'='*60}")
        self._log(f"Creating revenue-based features")
        self._log(f"{'='*60}")
        self._log(f"Current month: {self.MONTH_ABBREV[current_month]} (using months {current_month}-12)")
        self._log(f"Months included: {[self.MONTH_ABBREV[m] for m in months_to_include]}")
        self._log(f"Found {len(forecast_cols)} Forecast columns (out of {len(all_forecast_cols)} total):")
        for col in forecast_cols:
            self._log(f"  - {col}")
        
        # Helper function to sum all forecast columns for a filtered DataFrame (vectorized)
        def sum_forecast_values(filtered_df: pd.DataFrame) -> float:
            if filtered_df.empty or not forecast_cols:
                return 0.0
            # Vectorized sum: select columns, convert to numeric, sum all at once
            return filtered_df[forecast_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values.sum()
        
        # Filter by each Revenue_Category using boolean masks (more efficient)
        cat_col = df[revenue_category_col]
        committed_signed_df = df[cat_col == committed_signed_value]
        committed_unsigned_df = df[cat_col == committed_unsigned_value]
        wtd_pipeline_df = df[cat_col == wtd_pipeline_value]
        
        self._log(f"\nRows by Revenue_Category:")
        self._log(f"  {committed_signed_value}: {len(committed_signed_df)} rows")
        self._log(f"  {committed_unsigned_value}: {len(committed_unsigned_df)} rows")
        self._log(f"  {wtd_pipeline_value}: {len(wtd_pipeline_df)} rows")
        
        # Calculate sums
        committed_signed_sum = sum_forecast_values(committed_signed_df)
        committed_unsigned_sum = sum_forecast_values(committed_unsigned_df)
        wtd_pipeline_sum = sum_forecast_values(wtd_pipeline_df)
        total_sum = sum_forecast_values(df)
        
        self._log(f"\nForecast column sums (months {current_month}-12):")
        self._log(f"  Committed - Signed: {committed_signed_sum:,.2f}")
        self._log(f"  Committed - Unsigned: {committed_unsigned_sum:,.2f}")
        self._log(f"  Wtd. Pipeline: {wtd_pipeline_sum:,.2f}")
        self._log(f"  Total (all rows): {total_sum:,.2f}")
        
        # Calculate ratios (avoid division by zero)
        committed_signed_ratio = committed_signed_sum / total_sum if total_sum != 0 else 0.0
        committed_unsigned_ratio = committed_unsigned_sum / total_sum if total_sum != 0 else 0.0
        wtd_pipeline_ratio = wtd_pipeline_sum / total_sum if total_sum != 0 else 0.0
        total_committed = committed_signed_ratio + committed_unsigned_ratio
        wtd_pip_commit_ratio = wtd_pipeline_ratio / total_committed if total_committed != 0 else 0.0
        commit_sign_unsign_ratio = committed_signed_ratio / committed_unsigned_ratio if committed_unsigned_ratio != 0 else 0.0
        signed_vs_committed_ratio = committed_signed_ratio / total_committed if total_committed != 0 else 0.0
        
        result = {
            'Committed_Signed_revenue_ratio': committed_signed_ratio,
            'Committed_Unsigned_revenue_ratio': committed_unsigned_ratio,
            'Wtd_Pipeline_revenue_ratio': wtd_pipeline_ratio,
            'wtd_pip_commit_ratio': wtd_pip_commit_ratio,
            'commit_sign_unsign_ratio': commit_sign_unsign_ratio,
            'signed_vs_committed_ratio': signed_vs_committed_ratio,
            # Forecast sums by category (current month to Dec)
            'fcst_committed_signed_rem': committed_signed_sum,
            'fcst_committed_unsigned_rem': committed_unsigned_sum,
            'fcst_wtd_pipeline_rem': wtd_pipeline_sum
        }
        
        self._log(f"\n--- Revenue-Based Ratios ---")
        for key, value in list(result.items())[:6]:  # Only ratios (first 6)
            self._log(f"  {key}: {value:.4f} ({value*100:.2f}%)")
        
        self._log(f"\n--- Forecast Sums by Category (months {current_month}-12) ---")
        self._log(f"  fcst_committed_signed_rem: {committed_signed_sum:,.2f}")
        self._log(f"  fcst_committed_unsigned_rem: {committed_unsigned_sum:,.2f}")
        self._log(f"  fcst_wtd_pipeline_rem: {wtd_pipeline_sum:,.2f}")
        
        # --- Monthly Forecast Sums (all 12 months) - OPTIMIZED ---
        self._log(f"\n--- Monthly Forecast Sums ---")
        
        # Build column map once for all months (optimized)
        col_map = self._build_forecast_column_map(df)
        
        # Pre-compute numeric values for all mapped columns at once (vectorized)
        mapped_cols = [col_map[m] for m in range(1, 13) if m in col_map]
        if mapped_cols:
            # Convert all at once, then get sums per column
            numeric_df = df[mapped_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            col_sums = numeric_df.sum()
        else:
            col_sums = pd.Series()
        
        # Assign to result using pre-computed sums
        for month_num in range(1, 13):
            month_name = self.MONTH_ABBREV[month_num]
            if month_num in col_map:
                monthly_sum = col_sums[col_map[month_num]]
            else:
                monthly_sum = 0.0
            result[f'Forecast_{month_name}'] = monthly_sum
            self._log(f"  Forecast_{month_name}: {monthly_sum:,.2f}")
        
        # --- Forecast_yearly: Sum of all forecast columns Jan-Dec ---
        forecast_yearly = sum(result.get(f'Forecast_{self.MONTH_ABBREV[m]}', 0.0) for m in range(1, 13))
        result['Forecast_yearly'] = forecast_yearly
        self._log(f"\n--- Forecast Yearly (Jan-Dec) ---")
        self._log(f"  Forecast_yearly: {forecast_yearly:,.2f}")
        
        # --- revenue: Current month's forecast column sum ---
        current_month_name = self.MONTH_ABBREV[current_month]
        revenue = result.get(f'Forecast_{current_month_name}', 0.0)
        result['revenue'] = revenue
        self._log(f"\n--- Current Month Revenue ---")
        self._log(f"  revenue (Forecast-{current_month_name}): {revenue:,.2f}")
        
        # --- forecast_roy: Rest of Year (current month to Dec) ---
        forecast_roy = total_sum
        result['forecast_roy'] = forecast_roy
        self._log(f"\n--- Forecast ROY & Year-End Features ---")
        self._log(f"  forecast_roy (months {current_month}-12): {forecast_roy:,.2f}")
        
        # --- total_per_month_left: Average forecast per remaining month ---
        months_left = 12 - current_month + 1  # e.g., Oct=3 (Oct,Nov,Dec), Nov=2, Dec=1
        total_per_month_left = forecast_roy / months_left if months_left > 0 else 0.0
        result['total_per_month_left'] = total_per_month_left
        self._log(f"  months_left: {months_left}")
        self._log(f"  total_per_month_left (forecast_roy/{months_left}): {total_per_month_left:,.2f}")
        
        # --- year_end_rev and Forecast_gap (require YTD) ---
        if ytd is not None:
            year_end_rev = ytd + forecast_roy
            forecast_gap = forecast_roy - ytd
            result['year_end_rev'] = year_end_rev
            result['Forecast_gap'] = forecast_gap
            self._log(f"  YTD (provided): {ytd:,.2f}")
            self._log(f"  year_end_rev (YTD + forecast_roy): {year_end_rev:,.2f}")
            self._log(f"  Forecast_gap (forecast_roy - YTD): {forecast_gap:,.2f}")
        else:
            result['year_end_rev'] = 0.0
            result['Forecast_gap'] = 0.0
            self._log(f"  Warning: YTD not provided. year_end_rev and Forecast_gap set to 0.")
        
        # --- prev_month_achv_pct (requires historical_df) ---
        # Achievement % = actual revenue / forecasted revenue for previous month
        # NOTE: Handles year wrap-around (e.g., Jan 2025 -> Dec 2024)
        prev_month_achv_pct = 0.0
        
        if historical_df is not None and current_year is not None:
            # Handle year wrap-around for January
            if current_month == 1:
                prev_month = 12
                prev_month_year = current_year - 1
            else:
                prev_month = current_month - 1
                prev_month_year = current_year
            
            prev_month_abbrev = self.MONTH_ABBREV[prev_month]
            
            self._log(f"\n--- Previous Month Achievement % ---")
            self._log(f"  Looking up: {prev_month_year} {prev_month_abbrev} (month {prev_month})")
            
            # Find the row for previous month in historical data
            prev_month_row = historical_df[
                (historical_df[year_col] == prev_month_year) & 
                (historical_df[month_col] == prev_month)
            ]
            
            if len(prev_month_row) > 0:
                if len(prev_month_row) > 1:
                    self._log(f"  Warning: Multiple rows found, using first")
                
                # Get actual revenue for previous month
                actual_revenue = float(prev_month_row['revenue'].iloc[0]) if 'revenue' in prev_month_row.columns else 0.0
                
                # Find the Forecast column for previous month in historical data
                forecast_col_name = None
                for col in prev_month_row.columns:
                    col_lower = col.lower()
                    if col_lower.startswith('forecast') and f'-{prev_month_abbrev.lower()}' in col_lower:
                        forecast_col_name = col
                        break
                
                if forecast_col_name is None:
                    # Try alternative pattern
                    for col in prev_month_row.columns:
                        col_lower = col.lower()
                        if col_lower.startswith('forecast') and prev_month_abbrev.lower() in col_lower:
                            forecast_col_name = col
                            break
                
                if forecast_col_name:
                    forecasted_revenue = float(prev_month_row[forecast_col_name].iloc[0])
                    # Formula: ABS((Forecast / Actual) * 100)
                    if actual_revenue != 0:
                        prev_month_achv_pct = abs((forecasted_revenue / actual_revenue) * 100)
                    self._log(f"  Actual revenue ({prev_month_abbrev}): {actual_revenue:,.2f}")
                    self._log(f"  Forecasted revenue ({forecast_col_name}): {forecasted_revenue:,.2f}")
                    self._log(f"  prev_month_achv_pct: {prev_month_achv_pct:.2f}%")
                else:
                    self._log(f"  Warning: No Forecast-{prev_month_abbrev} column found in historical data")
            else:
                self._log(f"  Warning: No data found for {prev_month_year} month {prev_month}")
        else:
            self._log(f"\n--- Previous Month Achievement % ---")
            self._log(f"  Warning: historical_df or current_year not provided. prev_month_achv_pct set to 0.")
        
        result['prev_month_achv_pct'] = prev_month_achv_pct
        
        return result
    
    #-----------------------------------------------------------------------
    # Feature Creator: Calculate probability percentage statistics (mean/median)
    #-----------------------------------------------------------------------

    def create_prob_pct_features(
        self,
        df: pd.DataFrame,
        method: str = "mean",
        revenue_category_col: str = "Revenue_Category",
        prob_pct_col: str = "ProbabilityPer",
        wtd_pipeline_value: str = "Wtd. Pipeline"
    ) -> dict:
        """
        Create probability percentage features from refined raw data.
        
        Features calculated:
        - mean_prob_pct_wtd_pip: Statistical measure of ProbabilityPer where Revenue_Category is Wtd. Pipeline
        
        Args:
            df: DataFrame with refined raw data.
            method: Statistical method to use. Options: 'mean', 'median', 'mode', 'std', 'min', 'max', 'sum', 'var'.
            revenue_category_col: Column name for revenue category.
            prob_pct_col: Column name for probability percentage.
            wtd_pipeline_value: Value for Wtd. Pipeline in revenue_category.
            
        Returns:
            Dictionary with probability percentage features.
        """
        valid_methods = ['mean', 'median', 'mode', 'std', 'min', 'max', 'sum', 'var', 'count']
        
        if method.lower() not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Valid options: {valid_methods}")
        
        if revenue_category_col not in df.columns:
            raise ValueError(f"Column '{revenue_category_col}' not found in DataFrame.")
        
        if prob_pct_col not in df.columns:
            raise ValueError(f"Column '{prob_pct_col}' not found in DataFrame.")
        
        self._log(f"\n{'='*60}")
        self._log(f"Creating probability percentage features")
        self._log(f"{'='*60}")
        self._log(f"Method: {method}")
        
        # Filter to Wtd. Pipeline rows
        wtd_pipeline_df = df[df[revenue_category_col] == wtd_pipeline_value]
        
        self._log(f"Rows with {revenue_category_col}='{wtd_pipeline_value}': {len(wtd_pipeline_df)}")
        
        if len(wtd_pipeline_df) == 0:
            self._log(f"Warning: No rows found for {wtd_pipeline_value}")
            return {f'{method}_prob_pct_wtd_pip': 0.0}
        
        # Get the probability percentage values
        prob_values = pd.to_numeric(wtd_pipeline_df[prob_pct_col], errors='coerce').dropna()
        
        self._log(f"Valid {prob_pct_col} values: {len(prob_values)}")
        
        if len(prob_values) == 0:
            self._log(f"Warning: No valid numeric values in {prob_pct_col}")
            return {f'{method}_prob_pct_wtd_pip': 0.0}
        
        # Calculate the statistic using dictionary dispatch (faster than if-elif)
        method_lower = method.lower()
        
        method_dispatch = {
            'mean': lambda s: s.mean(),
            'median': lambda s: s.median(),
            'mode': lambda s: s.mode().iloc[0] if len(s.mode()) > 0 else 0.0,
            'std': lambda s: s.std(),
            'min': lambda s: s.min(),
            'max': lambda s: s.max(),
            'sum': lambda s: s.sum(),
            'var': lambda s: s.var(),
            'count': lambda s: len(s),
        }
        
        result_value = float(method_dispatch.get(method_lower, lambda s: 0.0)(prob_values))
        
        feature_name = f'{method_lower}_prob_pct_wtd_pip'
        
        result = {feature_name: result_value}
        
        self._log(f"\n--- Probability Percentage Features ---")
        self._log(f"  {feature_name}: {result_value:.4f}")
        
        # Only compute descriptive stats if verbose (lazy evaluation)
        if self.verbose:
            self._log(f"\n  Descriptive stats for {prob_pct_col} (Wtd. Pipeline):")
            self._log(f"    Count: {len(prob_values)}")
            self._log(f"    Mean: {prob_values.mean():.4f}")
            self._log(f"    Median: {prob_values.median():.4f}")
            self._log(f"    Std: {prob_values.std():.4f}")
            self._log(f"    Min: {prob_values.min():.4f}")
            self._log(f"    Max: {prob_values.max():.4f}")
        
        return result
    
    # -------------------------------------------------------------------------
    # Feature Creator: Generate time-based features (year and month integers)
    # -------------------------------------------------------------------------
    def create_time_features(
        self,
        current_year: int,
        current_month: int
    ) -> dict:
        """
        Create time-based features (year and month).
        
        Features calculated:
        - year: Current year as integer
        - month: Current month as integer (1-12)
        
        Args:
            current_year: Current year (e.g., 2025).
            current_month: Current month (1-12).
            
        Returns:
            Dictionary with year and month features.
        """
        if current_month < 1 or current_month > 12:
            raise ValueError(f"current_month must be 1-12, got {current_month}")
        
        self._log(f"\n{'='*60}")
        self._log(f"Creating time features")
        self._log(f"{'='*60}")
        
        result = {
            'year': current_year,
            'month': current_month,
            'month_id': f"{current_year}-{current_month:02d}"
        }
        
        self._log(f"  year: {current_year}")
        self._log(f"  month: {current_month} ({self.MONTH_ABBREV[current_month]})")
        
        return result    
    # -------------------------------------------------------------------------
    # Main Pipeline: End-to-end feature generation (fetch, refine, create)
    # -------------------------------------------------------------------------
    def get_all_features(
        self,
        prob_pct_method: str = "mean",
        historical_data_path: str = None
    ) -> pd.DataFrame:
        """
        End-to-end function to fetch data, refine it, and generate all features.
        
        This function handles:
        1. Fetching historical transformed data (if path provided)
        2. Fetching newly arrived raw data
        3. Refining the raw data
        4. Using current month/year from constructor or path
        5. Creating all features (time, lag, revenue-based, probability)
        6. Returning everything as a single-row DataFrame
        
        Args:
            prob_pct_method: Statistical method for probability features ('mean', 'median', etc.).
            historical_data_path: ABFSS path to historical data file. Uses constructor value if not provided.
            
        Returns:
            DataFrame with one row containing all features.
            
        Raises:
            ValueError: If current month/year cannot be determined.
        """
        self._log(f"\n{'#'*60}")
        self._log(f"# FEATURE GENERATION PIPELINE")
        self._log(f"{'#'*60}")
        
        # =====================================================================
        # STEP 1: Fetch Historical Transformed Data (optional)
        # =====================================================================
        hist_path = historical_data_path or self.historical_data_path
        historical_df = None
        
        if hist_path:
            self._log(f"\n{'='*60}")
            self._log(f"STEP 1: Fetching Historical Transformed Data")
            self._log(f"{'='*60}")
            
            historical_df = self.fetch_historical_transformed_data(path=hist_path)
            self._log(f"Historical data shape: {historical_df.shape}")
        else:
            self._log(f"\n{'='*60}")
            self._log(f"STEP 1: Skipping Historical Data (no path provided)")
            self._log(f"{'='*60}")
        
        # =====================================================================
        # STEP 2: Fetch Newly Arrived Raw Data
        # =====================================================================
        self._log(f"\n{'='*60}")
        self._log(f"STEP 2: Fetching Newly Arrived Raw Data")
        self._log(f"{'='*60}")
        
        raw_df = self.fetch_newly_arrived_raw_data()
        self._log(f"Raw data shape: {raw_df.shape}")
        
        # =====================================================================
        # STEP 3: Refine Raw Data
        # =====================================================================
        self._log(f"\n{'='*60}")
        self._log(f"STEP 3: Refining Raw Data")
        self._log(f"{'='*60}")
        
        refined_df = self.refine_raw_data(raw_df)
        self._log(f"Refined data shape: {refined_df.shape}")
        
        # =====================================================================
        # STEP 4: Get Current Month/Year
        # =====================================================================
        self._log(f"\n{'='*60}")
        self._log(f"STEP 4: Determining Current Month/Year")
        self._log(f"{'='*60}")
        
        current_month = self._current_month
        current_year = self._current_year
        
        if current_month is None or current_year is None:
            raise ValueError(
                "Could not determine current month/year. "
                "Please provide current_year and current_month in the constructor, "
                "or ensure the raw_data_path contains year/month info."
            )
        
        self._log(f"Current period: {current_year} {self.MONTH_ABBREV[current_month]} (month {current_month})")
        
        # =====================================================================
        # STEP 5: Generate All Features
        # =====================================================================
        self._log(f"\n{'='*60}")
        self._log(f"STEP 5: Generating Features")
        self._log(f"{'='*60}")
        
        # 5.1 Time features
        time_features = self.create_time_features(
            current_year=current_year,
            current_month=current_month
        )
        
        # 5.2 Lag features (need this first to get YTD)
        lag_features = self.create_lag_features(
            refined_df,
            current_month=current_month,
            current_year=current_year,
            historical_df=historical_df
        )
        
        # Get YTD from lag features
        ytd_value = lag_features.get('YTD', 0.0)
        
        # 5.3 Revenue-based features
        revenue_features = self.create_revenue_based_features(
            refined_df,
            current_month=current_month,
            current_year=current_year,
            ytd=ytd_value,
            historical_df=historical_df
        )
        
        # 5.4 Probability percentage features
        prob_pct_features = self.create_prob_pct_features(
            refined_df,
            method=prob_pct_method
        )
        
        # =====================================================================
        # STEP 6: Combine All Features into DataFrame
        # =====================================================================
        self._log(f"\n{'='*60}")
        self._log(f"STEP 6: Combining All Features")
        self._log(f"{'='*60}")
        
        # Combine all features into one dictionary
        all_features = {
            **time_features,
            **lag_features,
            **revenue_features,
            **prob_pct_features
        }
        
        # Convert to DataFrame (single row)
        features_df = pd.DataFrame([all_features])
        
        self._log(f"\n{'#'*60}")
        self._log(f"# FEATURE GENERATION COMPLETE")
        self._log(f"{'#'*60}")
        self._log(f"\nTotal features: {len(all_features)}")
        self._log(f"DataFrame shape: {features_df.shape}")
        self._log(f"\nFeature columns:")
        for i, col in enumerate(features_df.columns, 1):
            value = features_df[col].iloc[0]
            if isinstance(value, (int, float)):
                self._log(f"  {i:2d}. {col}: {value:,.4f}")
            else:
                self._log(f"  {i:2d}. {col}: {value}")
        
        return features_df
    
    # -------------------------------------------------------------------------
    # Helper: Sum values of a forecast column for a specific month
    # -------------------------------------------------------------------------
    def _sum_forecast_column(
        self,
        df: pd.DataFrame,
        month_num: int,
        label: str
    ) -> float:
        """
        Find the forecast column for a given month and sum its values.
        
        Uses cached column mapping for performance.
        
        Args:
            df: DataFrame (already filtered to Actuals).
            month_num: Month number (1-12).
            label: Label for logging (e.g., 'LM1').
            
        Returns:
            Sum of the forecast column values, or 0 if not found.
        """
        # Use cached column mapping
        col_map = self._build_forecast_column_map(df)
        
        if month_num not in col_map:
            month_abbrev = self.MONTH_ABBREV[month_num]
            self._log(f"  Warning: No forecast column found for {month_abbrev} ({label})")
            return 0.0
        
        col = col_map[month_num]
        total = df[col].sum()
        self._log(f"  {label}: Using column '{col}' → sum = {total:,.2f}")
        
        return float(total)
    
    # -------------------------------------------------------------------------
    # Helper: Calculate Year-to-Date sum of forecast columns (Jan to current-1)
    # -------------------------------------------------------------------------
    def _calculate_ytd(
        self,
        df: pd.DataFrame,
        current_month: int
    ) -> float:
        """
        Calculate Year-to-Date sum of all forecast columns up to (excluding) current month.
        
        Uses cached column mapping for performance.
        
        Args:
            df: DataFrame (already filtered to Actuals).
            current_month: Current month (1-12). YTD includes months 1 to current_month-1.
            
        Returns:
            YTD sum of forecast values.
        """
        if current_month == 1:
            self._log(f"  YTD: Current month is Jan, no previous months → YTD = 0")
            return 0.0
        
        # Use cached column mapping
        col_map = self._build_forecast_column_map(df)
        
        ytd_months = list(range(1, current_month))
        month_names = [self.MONTH_ABBREV[m] for m in ytd_months]
        self._log(f"  YTD: Summing forecast columns for months: {', '.join(month_names)}")
        
        # Vectorized sum using cached columns
        ytd_cols = [col_map[m] for m in ytd_months if m in col_map]
        
        if not ytd_cols:
            self._log(f"  Warning: No forecast columns found for YTD months")
            return 0.0
        
        # Sum all YTD columns at once (vectorized)
        ytd_total = df[ytd_cols].sum().sum()
        
        self._log(f"  YTD total ({len(ytd_cols)} columns): {ytd_total:,.2f}")
        
        return float(ytd_total)
    
    # -------------------------------------------------------------------------
    # Cache Manager: Clear all cached data (historical, raw, refined, indices)
    # -------------------------------------------------------------------------
    def clear_cache(self) -> None:
        """Clear all cached data including forecast column mapping and indices."""
        self._historical_data = None
        self._raw_data = None
        self._refined_data = None
        self._forecast_col_cache = None
        self._historical_index = None
        self._log("Cache cleared.")

##calling transformer here
###########################################

if __name__ == "__main__":
    # Define your ABFSS paths
    RAW_DATA_PATH = full_path
    
    # Optional: Path to historical data (if you have one)
    # HISTORICAL_DATA_PATH = "abfss://historicaldata@gtastorage.dfs.core.windows.net/data/Master_DF_RF_YTD_minus_Target.csv"
    HISTORICAL_DATA_PATH = "fpnacopilot.data_engineering.transform_data"  # Set to None if not available
    
    # Initialize the transformer with ABFSS paths
    # Year (2026) and month (Feb=2) are automatically extracted from the path
    transformer = DataTransformer(
        raw_data_path=RAW_DATA_PATH,
        historical_data_path=HISTORICAL_DATA_PATH,
        verbose=True
    )
    
    try:
        # Single function call to fetch data, refine, and generate all features
        features_df = transformer.get_all_features(
            prob_pct_method="mean"  # Options: 'mean', 'median', 'mode', 'std', 'min', 'max', 'sum', 'var', 'count'
        )
        
        # Display the final DataFrame
        print(f"\n{'#'*60}")
        print(f"# FINAL FEATURES DATAFRAME")
        print(f"{'#'*60}")
        print(features_df.T)  # Transpose for better readability
        
        # In Databricks, you can also use:
        display(features_df)
        
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


#####upsert on delta table 

from delta.tables import DeltaTable
from pyspark.sql import functions as F

# Create Delta table in catalog fpnacopilot with Column Mapping enabled
spark_df = spark.createDataFrame(features_df)

# Cast int64 columns to match existing table schema
int_cols = ["year", "month"]
for col in int_cols:
    spark_df = spark_df.withColumn(col, spark_df[col].cast("int"))

# Cast forecast and revenue columns to double
double_cols = ["Forecast_Jan", "Forecast_Feb", "Forecast_Mar", "Forecast_Apr", 
               "Forecast_May", "Forecast_Jun", "Forecast_Jul", "Forecast_Aug", 
               "Forecast_Sep", "Forecast_Oct", "Forecast_Nov", "Forecast_Dec",
               "Forecast_yearly", "revenue", "forecast_roy"]
for col in double_cols:
    spark_df = spark_df.withColumn(col, spark_df[col].cast("double"))

# Add last_updated_at column with current timestamp in IST timezone
spark_df = spark_df.withColumn(
    "last_updated_at",
    F.from_utc_timestamp(F.current_timestamp(), "Asia/Kolkata")
)

table_name = "fpnacopilot.data_engineering.transform_data"

# Prepare DeltaTable object
delta_table = DeltaTable.forName(spark, table_name)

# Define merge condition on year and month
merge_condition = "target.year = source.year AND target.month = source.month"

# Prepare update and insert mappings for all columns, set last_updated_at to current IST timestamp
update_set = {col: f"source.{col}" for col in spark_df.columns}
insert_set = {col: f"source.{col}" for col in spark_df.columns}

# Perform upsert (merge) operation
(
    delta_table.alias("target")
    .merge(
        spark_df.alias("source"),
        merge_condition
    )
    .whenMatchedUpdate(set=update_set)
    .whenNotMatchedInsert(values=insert_set)
    .execute()
)

display(spark.read.table(table_name).toPandas().shape)
display(spark.read.table(table_name).toPandas())


