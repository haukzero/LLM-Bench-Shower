import sqlite3
import json
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from contextlib import contextmanager


class BenchmarkDatabase:
    def __init__(self, db_path: str = None):
        if db_path is None:
            # Default to backend/data/benchmark_results.db
            backend_dir = Path(__file__).parent
            data_dir = backend_dir / "data"
            data_dir.mkdir(exist_ok=True)
            db_path = str(data_dir / "benchmark_results.db")
        else:
            # Ensure the directory for the provided path exists
            db_path_obj = Path(db_path)
            db_path_obj.parent.mkdir(parents=True, exist_ok=True)

        self.db_path = db_path
        self._init_database()

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_database(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Create benchmark_results table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS benchmark_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    dataset_name TEXT NOT NULL,
                    results_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(model_name, dataset_name)
                )
            """
            )

            # Create index for faster lookups
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_model_dataset 
                ON benchmark_results(model_name, dataset_name)
            """
            )

            # Create trigger to update updated_at timestamp
            cursor.execute(
                """
                CREATE TRIGGER IF NOT EXISTS update_timestamp 
                AFTER UPDATE ON benchmark_results
                FOR EACH ROW
                BEGIN
                    UPDATE benchmark_results 
                    SET updated_at = CURRENT_TIMESTAMP
                    WHERE id = NEW.id;
                END
            """
            )

    def save_result(self, model_name: str, dataset_name: str, results: Dict) -> None:
        """Save or update a benchmark result.

        Args:
            model_name: Name of the model
            dataset_name: Name of the dataset
            results: Benchmark results dictionary
        """
        results_json = json.dumps(results, ensure_ascii=False)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO benchmark_results (model_name, dataset_name, results_json)
                VALUES (?, ?, ?)
                ON CONFLICT(model_name, dataset_name) 
                DO UPDATE SET 
                    results_json = excluded.results_json,
                    updated_at = CURRENT_TIMESTAMP
            """,
                (model_name, dataset_name, results_json),
            )

    def save_results_batch(self, results_list: List[Tuple[str, str, Dict]]) -> None:
        """Save or update multiple benchmark results in a single transaction.

        Args:
            results_list: List of tuples (model_name, dataset_name, results)
        """
        if not results_list:
            return

        with self._get_connection() as conn:
            cursor = conn.cursor()
            for model_name, dataset_name, results in results_list:
                results_json = json.dumps(results, ensure_ascii=False)
                cursor.execute(
                    """
                    INSERT INTO benchmark_results (model_name, dataset_name, results_json)
                    VALUES (?, ?, ?)
                    ON CONFLICT(model_name, dataset_name) 
                    DO UPDATE SET 
                        results_json = excluded.results_json,
                        updated_at = CURRENT_TIMESTAMP
                """,
                    (model_name, dataset_name, results_json),
                )

    def get_result(self, model_name: str, dataset_name: str) -> Optional[Dict]:
        """Retrieve a benchmark result.

        Args:
            model_name: Name of the model
            dataset_name: Name of the dataset

        Returns:
            Benchmark results dictionary or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT results_json FROM benchmark_results
                WHERE model_name = ? AND dataset_name = ?
            """,
                (model_name, dataset_name),
            )

            row = cursor.fetchone()
            if row:
                return json.loads(row["results_json"])
            return None

    def get_all_results(self) -> List[Tuple[str, str, Dict, str, str]]:
        """Retrieve all benchmark results.

        Returns:
            List of tuples (model_name, dataset_name, results, created_at, updated_at)
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT model_name, dataset_name, results_json, created_at, updated_at
                FROM benchmark_results
                ORDER BY updated_at DESC
            """
            )

            results = []
            for row in cursor.fetchall():
                results.append(
                    (
                        row["model_name"],
                        row["dataset_name"],
                        json.loads(row["results_json"]),
                        row["created_at"],
                        row["updated_at"],
                    )
                )
            return results

    def get_results_by_model(self, model_name: str) -> List[Tuple[str, Dict, str, str]]:
        """Retrieve all results for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            List of tuples (dataset_name, results, created_at, updated_at)
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT dataset_name, results_json, created_at, updated_at
                FROM benchmark_results
                WHERE model_name = ?
                ORDER BY updated_at DESC
            """,
                (model_name,),
            )

            results = []
            for row in cursor.fetchall():
                results.append(
                    (
                        row["dataset_name"],
                        json.loads(row["results_json"]),
                        row["created_at"],
                        row["updated_at"],
                    )
                )
            return results

    def get_results_by_dataset(
        self, dataset_name: str
    ) -> List[Tuple[str, Dict, str, str]]:
        """Retrieve all results for a specific dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            List of tuples (model_name, results, created_at, updated_at)
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT model_name, results_json, created_at, updated_at
                FROM benchmark_results
                WHERE dataset_name = ?
                ORDER BY updated_at DESC
            """,
                (dataset_name,),
            )

            results = []
            for row in cursor.fetchall():
                results.append(
                    (
                        row["model_name"],
                        json.loads(row["results_json"]),
                        row["created_at"],
                        row["updated_at"],
                    )
                )
            return results

    def delete_result(self, model_name: str, dataset_name: str) -> bool:
        """Delete a benchmark result.

        Args:
            model_name: Name of the model
            dataset_name: Name of the dataset

        Returns:
            True if deleted, False if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                DELETE FROM benchmark_results
                WHERE model_name = ? AND dataset_name = ?
            """,
                (model_name, dataset_name),
            )
            return cursor.rowcount > 0

    def clear_all_results(self) -> int:
        """Clear all benchmark results.

        Returns:
            Number of rows deleted
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM benchmark_results")
            return cursor.rowcount

    def get_stats(self) -> Dict:
        """Get database statistics.

        Returns:
            Dictionary with statistics
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Total results
            cursor.execute("SELECT COUNT(*) as count FROM benchmark_results")
            total_results = cursor.fetchone()["count"]

            # Unique models
            cursor.execute(
                "SELECT COUNT(DISTINCT model_name) as count FROM benchmark_results"
            )
            unique_models = cursor.fetchone()["count"]

            # Unique datasets
            cursor.execute(
                "SELECT COUNT(DISTINCT dataset_name) as count FROM benchmark_results"
            )
            unique_datasets = cursor.fetchone()["count"]

            # Most recent update
            cursor.execute(
                "SELECT MAX(updated_at) as last_update FROM benchmark_results"
            )
            last_update = cursor.fetchone()["last_update"]

            return {
                "total_results": total_results,
                "unique_models": unique_models,
                "unique_datasets": unique_datasets,
                "last_update": last_update,
                "database_path": self.db_path,
            }
