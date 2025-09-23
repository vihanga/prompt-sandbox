"""
Result storage and retrieval system
"""

import json
import sqlite3
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from .runner import ExperimentResult


class ResultStorage:
    """
    Persistent storage for experiment results

    Supports both JSON (simple, portable) and SQLite (queryable) backends.
    """

    def __init__(self, storage_dir: Path, use_sqlite: bool = False):
        """
        Initialize result storage

        Args:
            storage_dir: Directory for storing results
            use_sqlite: Use SQLite database instead of JSON files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.use_sqlite = use_sqlite

        if use_sqlite:
            self.db_path = self.storage_dir / "experiments.db"
            self._init_database()

    def _init_database(self):
        """Initialize SQLite database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_name TEXT NOT NULL,
                prompt_name TEXT NOT NULL,
                model_name TEXT NOT NULL,
                test_case_id INTEGER NOT NULL,
                generated_text TEXT,
                reference_text TEXT,
                generation_time REAL,
                timestamp REAL NOT NULL,
                input_data TEXT,
                evaluation_scores TEXT
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_experiment_name
            ON experiments(experiment_name)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_model_name
            ON experiments(model_name)
        """)

        conn.commit()
        conn.close()

    def save_results(
        self,
        experiment_name: str,
        results: List[ExperimentResult],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Save experiment results

        Args:
            experiment_name: Name of experiment
            results: List of ExperimentResult objects
            metadata: Optional metadata about experiment
        """
        if self.use_sqlite:
            self._save_to_sqlite(experiment_name, results)
        else:
            self._save_to_json(experiment_name, results, metadata)

    def _save_to_json(
        self,
        experiment_name: str,
        results: List[ExperimentResult],
        metadata: Optional[Dict[str, Any]]
    ):
        """Save to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_name}_{timestamp}.json"
        filepath = self.storage_dir / filename

        data = {
            "experiment_name": experiment_name,
            "metadata": metadata or {},
            "num_results": len(results),
            "results": [self._result_to_dict(r) for r in results]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"💾 Results saved to: {filepath}")

    def _save_to_sqlite(
        self,
        experiment_name: str,
        results: List[ExperimentResult]
    ):
        """Save to SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for result in results:
            cursor.execute("""
                INSERT INTO experiments (
                    experiment_name, prompt_name, model_name, test_case_id,
                    generated_text, reference_text, generation_time,
                    timestamp, input_data, evaluation_scores
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.experiment_name,
                result.prompt_name,
                result.model_name,
                result.test_case_id,
                result.generated_text,
                result.reference_text,
                result.generation_time,
                result.timestamp,
                json.dumps(result.input_data),
                json.dumps(result.evaluation_scores)
            ))

        conn.commit()
        conn.close()

        print(f"💾 Saved {len(results)} results to SQLite: {self.db_path}")

    def load_results(
        self,
        experiment_name: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Load experiment results

        Args:
            experiment_name: Filter by experiment name (None = all)
            limit: Maximum number of results to return

        Returns:
            List of result dictionaries
        """
        if self.use_sqlite:
            return self._load_from_sqlite(experiment_name, limit)
        else:
            return self._load_from_json(experiment_name, limit)

    def _load_from_json(
        self,
        experiment_name: Optional[str],
        limit: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Load from JSON files"""
        results = []

        # Find matching JSON files
        pattern = f"{experiment_name}_*.json" if experiment_name else "*.json"
        files = sorted(self.storage_dir.glob(pattern), reverse=True)

        if limit:
            files = files[:limit]

        for filepath in files:
            with open(filepath, 'r') as f:
                data = json.load(f)
                results.extend(data.get("results", []))

        return results

    def _load_from_sqlite(
        self,
        experiment_name: Optional[str],
        limit: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Load from SQLite database"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM experiments"
        params = []

        if experiment_name:
            query += " WHERE experiment_name = ?"
            params.append(experiment_name)

        query += " ORDER BY timestamp DESC"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        results = []
        for row in rows:
            results.append({
                "experiment_name": row["experiment_name"],
                "prompt_name": row["prompt_name"],
                "model_name": row["model_name"],
                "test_case_id": row["test_case_id"],
                "generated_text": row["generated_text"],
                "reference_text": row["reference_text"],
                "generation_time": row["generation_time"],
                "timestamp": row["timestamp"],
                "input_data": json.loads(row["input_data"]),
                "evaluation_scores": json.loads(row["evaluation_scores"])
            })

        conn.close()
        return results

    def _result_to_dict(self, result: ExperimentResult) -> Dict[str, Any]:
        """Convert ExperimentResult to dictionary"""
        return {
            "experiment_name": result.experiment_name,
            "prompt_name": result.prompt_name,
            "model_name": result.model_name,
            "test_case_id": result.test_case_id,
            "input_data": result.input_data,
            "generated_text": result.generated_text,
            "reference_text": result.reference_text,
            "generation_time": result.generation_time,
            "evaluation_scores": result.evaluation_scores,
            "timestamp": result.timestamp
        }
