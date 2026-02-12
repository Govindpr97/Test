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
