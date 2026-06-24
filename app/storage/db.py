import sqlite3
import json
import os
from datetime import datetime
from app.utils.logger import logger

DB_PATH = "artifacts/sage_experiments.db"
os.makedirs("artifacts", exist_ok=True)

def get_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    conn=get_connection()
    cursor=conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            dataset_name TEXT,
            target_column TEXT,
            task_type TEXT,
            best_model TEXT,
            n_features INTEGER,
            n_rows INTEGER,
            untuned_metrics TEXT,
            tuned_metrics TEXT,
            best_params TEXT,
            dropped_columns TEXT,
            preprocessing_steps INTEGER,
            model_path TEXT,
            notes TEXT
        )
    """)

    conn.commit()
    conn.close()
    logger.info("Databaase Connected")

def log_experiment(
    dataset_name: str,
    target_column: str,
    task_type: str,
    best_model: str,
    n_features: int,
    n_rows: int,
    untuned_metrics: dict,
    tuned_metrics: dict,
    best_params: dict,
    dropped_columns: list,
    preprocessing_steps: int,
    model_path: str,
    notes: str = ""
) -> str:
    
    try:
        init_db()
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO experiments (
                run_id, timestamp, dataset_name, target_column,
                task_type, best_model, n_features, n_rows,
                untuned_metrics, tuned_metrics, best_params,
                dropped_columns, preprocessing_steps, model_path, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id, timestamp, dataset_name, target_column,
            task_type, best_model, n_features, n_rows,
            json.dumps(untuned_metrics), json.dumps(tuned_metrics),
            json.dumps(best_params), json.dumps(dropped_columns),
            preprocessing_steps, model_path, notes
        ))
        conn.commit()
        conn.close()
        logger.info(f"Experiment logged: {run_id}")
        return run_id

    except Exception as e:
        logger.error(f"Failed to log experiment: {e}")
        raise

def get_all_experiments() -> list[dict]:
    try:
        init_db()
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM experiments ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        conn.close()

        experiments = []
        for row in rows:
            exp = dict(zip(columns, row))
            for field in ["untuned_metrics", "tuned_metrics", "best_params", "dropped_columns"]:
                try:
                    exp[field] = json.loads(exp[field])
                except Exception:
                    pass
            experiments.append(exp)

        return experiments

    except Exception as e:
        logger.error(f"Failed to fetch experiments: {e}")
        return []


def delete_experiment(run_id: str):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM experiments WHERE run_id = ?", (run_id,))
        conn.commit()
        conn.close()
        logger.info(f"Experiment deleted: {run_id}")
    except Exception as e:
        logger.error(f"Failed to delete experiment: {e}")
        raise

