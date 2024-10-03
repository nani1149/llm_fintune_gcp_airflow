from __future__ import annotations

import os
from datetime import datetime
import importlib

from google.cloud.aiplatform import schema
from google.protobuf.json_format import ParseDict
from google.protobuf.struct_pb2 import Value
from google.cloud import aiplatform
from airflow.operators.python import PythonOperator
import uuid
from ..include.llm_finetune import create_llama3_fintune

from airflow.models.dag import DAG


# common_util = importlib.import_module(
#     "vertex-ai-samples.community-content.vertex_model_garden.model_oss.notebook_util.common_util"
# )

DAG_ID = "vertex_ai_llm_finetune_operations"


with DAG(
    f"{DAG_ID}_custom_container",
    schedule="@once",
    start_date=datetime(2021, 1, 1),
    catchup=False,
    tags=["experiment", "vertex_ai", "llm_finetune"],
) as dag:
    run_this = PythonOperator(task_id="print_the_context", python_callable=create_llama3_fintune)
    (run_this)

    