source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
SPECULATIVE_CONFIG=$(mtp_spec_config 5)

nic_name="$NIC_NAME"
local_ip="$D1_IP" # change to your own ip

# External DP requires an API server for every rank.
server_role_args="--api-server-count 1"

export HCCL_OP_EXPANSION_MODE="AIV"

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

#Mooncake
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export TASK_QUEUE_ENABLE=1
export ASCEND_RT_VISIBLE_DEVICES=$1

export VLLM_ASCEND_ENABLE_MLAPO=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

export HCCL_INTER_HCCS_DISABLE=FALSE
export HCCL_CONCURRENT_ENABLE=0

export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=1091
export HCCL_EVENT_TIMEOUT=1200
export ACL_STREAM_TIMEOUT=1300000
export ACL_DEVICE_SYNC_TIMEOUT=1300

export ASCEND_PROCESS_LOG_PATH="$ROUND_DIR/log/ascend/decode-1-rank-$4"
mkdir -p "${ASCEND_PROCESS_LOG_PATH}"

case "$PERF_MODE" in
base)
    MAX_MODEL_LEN=202752
    export HCCL_BUFFSIZE=256
    export VLLM_ASCEND_ENABLE_FUSED_MC2=1
    VLLM_ADDITIONAL_CONFIG='{"fuse_muls_add":true, "recompute_scheduler_enable":true, "enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true}'
    ;;
optimized)
    MAX_MODEL_LEN=266240
    export HCCL_BUFFSIZE=640
    export VLLM_ASCEND_ENABLE_FUSED_MC2=0
    VLLM_ADDITIONAL_CONFIG='{"fuse_muls_add":true, "recompute_scheduler_enable":false, "enable_sparse_sfa_c8": '"$OPTIMIZED_D_SFA_C8"', "enable_sparse_li_c8": true, "multistream_overlap_shared_expert": true, "enable_shared_expert_dp": true, "enable_indexer_projection_reuse": true, "enable_fused_indexer_cache_write": true}'
    ;;
*) echo "unsupported PERF_MODE: $PERF_MODE (expected: base or optimized)" >&2; exit 2 ;;
esac

nohup vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port $2 \
    --data-parallel-size $3 \
    --data-parallel-rank $4 \
    --data-parallel-address $5 \
    --data-parallel-rpc-port $6 \
    --distributed-timeout-seconds 7200 \
    --cpu-distributed-timeout-seconds 7200 \
    --tensor-parallel-size $7 \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name glm-5 \
    --max-model-len "$MAX_MODEL_LEN" \
    --safetensors-load-strategy 'prefetch' \
    --max-num-batched-tokens 192 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --speculative-config "$SPECULATIVE_CONFIG" \
    --additional-config "$VLLM_ADDITIONAL_CONFIG" \
    --trust-remote-code \
    --max-num-seqs 32 \
    --gpu-memory-utilization 0.90 \
    ${server_role_args} \
    --async-scheduling \
    --quantization ascend \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --kv-transfer-config '{
        "kv_connector": "MooncakeConnectorV1",
        "kv_role": "kv_consumer",
        "kv_port": "30200",
        "engine_id": "2",
        "kv_connector_extra_config": {
            "use_ascend_direct": true,
            "prefill": {"dp_size": 1, "pp_size": 2, "tp_size": 16, "pp_layer_partition": "41,37"},
            "decode": { "dp_size": '"$3"', "tp_size": '"$7"'}
        }
    }' \
    > "$ROUND_DIR/log/decode-1-rank-$4.log" 2>&1 &
