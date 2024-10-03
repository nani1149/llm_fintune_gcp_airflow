import os
import yaml
from google.cloud import aiplatform



def load_config(file_path="config.yaml"):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)

def create_llama3_fintune():
    config = load_config()

    training_config = config['tasks']['training']

    BUCKET_URI = training_config['bucket_uri']
    STAGING_BUCKET = os.path.join(BUCKET_URI, "temporal")
    MODEL_BUCKET = os.path.join(BUCKET_URI, "llama3")
    project_number = training_config['project_number']
    SERVICE_ACCOUNT = f"{project_number}-compute@developer.gserviceaccount.com"
    HF_TOKEN = os.environ.get("HF_TOKEN", training_config['hf_token'])
    
    template = training_config['template']
    train_dataset_name = training_config['train_dataset_name']
    train_split_name = training_config['train_split_name']
    eval_dataset_name = training_config['eval_dataset_name']
    eval_split_name = training_config['eval_split_name']
    instruct_column_in_dataset = training_config['instruct_column_in_dataset']
    
    TRAIN_DOCKER_URI = training_config['train_docker_uri']
    MODEL_ID = training_config['model_id']
    accelerator_type = training_config['accelerator_type']
    per_device_train_batch_size = training_config['per_device_train_batch_size']
    gradient_accumulation_steps = training_config['gradient_accumulation_steps']
    max_seq_length = training_config['max_seq_length']
    max_steps = training_config['max_steps']
    num_epochs = training_config['num_epochs']
    finetuning_precision_mode = training_config['finetuning_precision_mode']
    learning_rate = training_config['learning_rate']
    lr_scheduler_type = training_config['lr_scheduler_type']
    
    lora_rank = training_config['lora_rank']
    lora_alpha = training_config['lora_alpha']
    lora_dropout = training_config['lora_dropout']
    
    enable_gradient_checkpointing = training_config['enable_gradient_checkpointing']
    attn_implementation = training_config['attn_implementation']
    optimizer = training_config['optimizer']
    warmup_ratio = training_config['warmup_ratio']
    report_to = training_config['report_to']
    
    save_steps = training_config['save_steps']
    logging_steps = training_config['logging_steps']
    
    machine_type = training_config['machine_type']
    replica_count = training_config['replica_count']
    
    job_name = training_config['job_name']
    base_output_dir = os.path.join(STAGING_BUCKET, job_name)
    
    lora_output_dir = os.path.join(base_output_dir, "adapter")
    merged_model_output_dir = os.path.join(base_output_dir, "merged-model")
    
    eval_args = [
        f"--eval_dataset_path={eval_dataset_name}",
        f"--eval_column={instruct_column_in_dataset}",
        f"--eval_template={template}",
        f"--eval_split={eval_split_name}",
        f"--eval_steps={save_steps}",
        "--eval_tasks=builtin_eval",
        "--eval_metric_name=loss",
    ]
    
    train_job_args = [
        "--config_file=vertex_vision_model_garden_peft/deepspeed_zero2_4gpu.yaml",
        "--task=instruct-lora",
        "--completion_only=True",
        f"--pretrained_model_id={MODEL_ID}",
        f"--dataset_name={train_dataset_name}",
        f"--train_split_name={train_split_name}",
        f"--instruct_column_in_dataset={instruct_column_in_dataset}",
        f"--output_dir={lora_output_dir}",
        f"--merge_base_and_lora_output_dir={merged_model_output_dir}",
        f"--per_device_train_batch_size={per_device_train_batch_size}",
        f"--gradient_accumulation_steps={gradient_accumulation_steps}",
        f"--lora_rank={lora_rank}",
        f"--lora_alpha={lora_alpha}",
        f"--lora_dropout={lora_dropout}",
        f"--max_steps={max_steps}",
        f"--max_seq_length={max_seq_length}",
        f"--learning_rate={learning_rate}",
        f"--lr_scheduler_type={lr_scheduler_type}",
        f"--precision_mode={finetuning_precision_mode}",
        f"--enable_gradient_checkpointing={enable_gradient_checkpointing}",
        f"--num_epochs={num_epochs}",
        f"--attn_implementation={attn_implementation}",
        f"--optimizer={optimizer}",
        f"--warmup_ratio={warmup_ratio}",
        f"--report_to={report_to}",
        f"--logging_output_dir={base_output_dir}",
        f"--save_steps={save_steps}",
        f"--logging_steps={logging_steps}",
        f"--template={template}",
        f"--huggingface_access_token={HF_TOKEN}",
    ] + eval_args
    
    train_job = aiplatform.CustomContainerTrainingJob(
        display_name=job_name,
        staging_bucket=STAGING_BUCKET,
        container_uri=TRAIN_DOCKER_URI,
    )
    
    train_job.run(
        args=train_job_args,
        environment_variables={"WANDB_DISABLED": True},
        replica_count=replica_count,
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=1,  # Assuming the count is always 1
        boot_disk_size_gb=500,
        service_account=SERVICE_ACCOUNT,
        base_output_dir=base_output_dir,
    )


def run_inference():
    config = load_config()

    inference_config = config['tasks']['inference']
    
    inference_model_id = inference_config['inference_model_id']
    inference_accelerator_type = inference_config['inference_accelerator_type']
    inference_machine_type = inference_config['inference_machine_type']
    inference_batch_size = inference_config['inference_batch_size']
    inference_max_seq_length = inference_config['inference_max_seq_length']
    inference_precision_mode = inference_config['inference_precision_mode']
    inference_model_output_dir = inference_config['inference_model_output_dir']

    # Code to run inference using the above parameters
    # Example (this depends on the exact inference framework):
    print(f"Running inference with model {inference_model_id}")
    # Inference logic here...
