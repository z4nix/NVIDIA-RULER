#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# container: docker.io/cphsieh/ruler:0.1.0
# bash run.sh MODEL_NAME BENCHMARK_NAME

if [ $# -ne 2 ]; then
    echo "Usage: $0 <model_name> <benchmark_name>"
    exit 1
fi

# Root Directories
SCRIPT_DIR=$(realpath $(dirname "$0")) # Absolute path to the script directory
GPUS="1" # GPU size for tensor_parallel
ROOT_DIR="${SCRIPT_DIR}/benchmark_root" # Path for generated task samples and model predictions
MODEL_DIR="${SCRIPT_DIR}/../.." # Path for model folders from Huggingface
ENGINE_DIR="${SCRIPT_DIR}" # Path for engine folders from TensorRT-LLM
BATCH_SIZE=1  # Increase to improve GPU utilization

# Model and Tokenizer
source "$SCRIPT_DIR/config_models.sh"
MODEL_NAME=${1}
MODEL_CONFIG=$(MODEL_SELECT ${MODEL_NAME} ${MODEL_DIR} ${ENGINE_DIR})
IFS=":" read MODEL_PATH MODEL_TEMPLATE_TYPE MODEL_FRAMEWORK TOKENIZER_PATH TOKENIZER_TYPE OPENAI_API_KEY GEMINI_API_KEY AZURE_ID AZURE_SECRET AZURE_ENDPOINT <<< "$MODEL_CONFIG"

if [ -z "${MODEL_PATH}" ]; then
    echo "Model: ${MODEL_NAME} is not supported"
    exit 1
fi

export OPENAI_API_KEY=${OPENAI_API_KEY}
export GEMINI_API_KEY=${GEMINI_API_KEY}
export AZURE_API_ID=${AZURE_ID}
export AZURE_API_SECRET=${AZURE_SECRET}
export AZURE_API_ENDPOINT=${AZURE_ENDPOINT}

# Benchmark and Tasks
source "$SCRIPT_DIR/config_tasks.sh"
BENCHMARK=${2}
declare -n TASKS=$BENCHMARK
if [ -z "${TASKS}" ]; then
    echo "Benchmark: ${BENCHMARK} is not supported"
    exit 1
fi

# Start server (optional for container environments)
if [ "$MODEL_FRAMEWORK" == "vllm" ]; then
    python "$SCRIPT_DIR/pred/serve_vllm.py" \
        --model=${MODEL_PATH} \
        --tensor-parallel-size=${GPUS} \
        --dtype bfloat16 \
        --disable-custom-all-reduce &
elif [ "$MODEL_FRAMEWORK" == "trtllm" ]; then
    python "$SCRIPT_DIR/pred/serve_trt.py" \
        --model_path=${MODEL_PATH} &
elif [ "$MODEL_FRAMEWORK" == "sglang" ]; then
    python -m sglang.launch_server \
        --model-path ${MODEL_PATH} \
        --tp ${GPUS} \
        --port 5000 \
        --enable-flashinfer &
fi

# Start client (prepare data / call model API / obtain final metrics)
total_time=0
for MAX_SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do
    RESULTS_DIR="${ROOT_DIR}/${MODEL_NAME}/${BENCHMARK}/${MAX_SEQ_LENGTH}"
    DATA_DIR="${RESULTS_DIR}/data"
    PRED_DIR="${RESULTS_DIR}/pred"
    mkdir -p "${DATA_DIR}"
    mkdir -p "${PRED_DIR}"
    
    for TASK in "${TASKS[@]}"; do
        python "$SCRIPT_DIR/data/prepare.py" \
            --save_dir "${DATA_DIR}" \
            --benchmark "${BENCHMARK}" \
            --task "${TASK}" \
            --tokenizer_path "${TOKENIZER_PATH}" \
            --tokenizer_type "${TOKENIZER_TYPE}" \
            --max_seq_length "${MAX_SEQ_LENGTH}" \
            --model_template_type "${MODEL_TEMPLATE_TYPE}" \
            --num_samples "${NUM_SAMPLES}" \
            ${REMOVE_NEWLINE_TAB}
        
        start_time=$(date +%s)
        python "$SCRIPT_DIR/pred/call_api.py" \
            --data_dir "${DATA_DIR}" \
            --save_dir "${PRED_DIR}" \
            --benchmark "${BENCHMARK}" \
            --task "${TASK}" \
            --server_type "${MODEL_FRAMEWORK}" \
            --model_name_or_path "${MODEL_PATH}" \
            --temperature "${TEMPERATURE}" \
            --top_k "${TOP_K}" \
            --top_p "${TOP_P}" \
            --batch_size "${BATCH_SIZE}" \
            ${STOP_WORDS}
        end_time=$(date +%s)
        time_diff=$((end_time - start_time))
        total_time=$((total_time + time_diff))
    done
    
    python "$SCRIPT_DIR/eval/evaluate.py" \
        --data_dir "${PRED_DIR}" \
        --benchmark "${BENCHMARK}"
done

echo "Total time spent on call_api: $total_time seconds"
