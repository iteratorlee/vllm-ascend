#!/usr/bin/env bash

MODEL_PATH=/public-flash/models/GLM-5.2-w8a8
P0_IP=172.16.10.181
P1_IP=172.16.10.182
D0_IP=172.16.10.183
D1_IP=172.16.10.184
NIC_NAME=business
MTP_MODEL_PATH=${MTP_MODEL_PATH:-}
if [[ -n "$MTP_MODEL_PATH" ]]; then
    MTP_VIEW_MODE=${MTP_VIEW_MODE:-loader}
else
    MTP_VIEW_MODE=${MTP_VIEW_MODE:-direct}
fi
[[ "$MTP_VIEW_MODE" == direct || "$MTP_VIEW_MODE" == loader ]] || {
    echo "MTP_VIEW_MODE must be direct or loader" >&2
    return 2
}
PREFILL_KV_PORT=${PREFILL_KV_PORT:-31000}
OPTIMIZED_D_SFA_C8=${OPTIMIZED_D_SFA_C8:-true}
OPTIMIZED_P_SFA_C8=${OPTIMIZED_P_SFA_C8:-true}
[[ "$OPTIMIZED_D_SFA_C8" == true || "$OPTIMIZED_D_SFA_C8" == false ]] || {
    echo "OPTIMIZED_D_SFA_C8 must be true or false" >&2
    return 2
}
[[ "$OPTIMIZED_P_SFA_C8" == true || "$OPTIMIZED_P_SFA_C8" == false ]] || {
    echo "OPTIMIZED_P_SFA_C8 must be true or false" >&2
    return 2
}

mtp_spec_config() {
    local tokens=${1:?missing speculative token count}
    if [[ "$MTP_VIEW_MODE" == loader ]]; then
        [[ -n "$MTP_MODEL_PATH" ]] || {
            echo "MTP_MODEL_PATH is required for loader view mode" >&2
            return 2
        }
        printf '{"num_speculative_tokens":%s,"method":"deepseek_mtp","enforce_eager":true}\n' "$tokens"
        return
    fi
    if [[ -n "$MTP_MODEL_PATH" ]]; then
        [[ "$MTP_MODEL_PATH" =~ ^/[A-Za-z0-9/._-]+$ ]] || {
            echo "MTP_MODEL_PATH contains unsupported characters" >&2
            return 2
        }
        printf '{"num_speculative_tokens":%s,"method":"deepseek_mtp","enforce_eager":true,"model":"%s","quantization":"ascend"}\n' "$tokens" "$MTP_MODEL_PATH"
    else
        printf '{"num_speculative_tokens":%s,"method":"deepseek_mtp","enforce_eager":true}\n' "$tokens"
    fi
}

PERF_MODE=${PERF_MODE:-optimized}
if [[ "$PERF_MODE" == optimized && "$OPTIMIZED_P_SFA_C8" != "$OPTIMIZED_D_SFA_C8" ]]; then
    echo "P/D sparse SFA C8 must match for Mooncake KV cache layout" >&2
    return 2
fi
DECODE_SCENE=${DECODE_SCENE:-dp2tp16}
case "$DECODE_SCENE" in
    dp32tp1) DECODE_DP_SIZE=32; DECODE_TP_SIZE=1 ;;
    dp16tp2) DECODE_DP_SIZE=16; DECODE_TP_SIZE=2 ;;
    dp8tp4) DECODE_DP_SIZE=8; DECODE_TP_SIZE=4 ;;
    dp4tp8) DECODE_DP_SIZE=4; DECODE_TP_SIZE=8 ;;
    dp2tp16) DECODE_DP_SIZE=2; DECODE_TP_SIZE=16 ;;
    *) echo "unsupported DECODE_SCENE: $DECODE_SCENE" >&2; return 2 ;;
esac

DECODE_DP_SIZE_LOCAL=$((DECODE_DP_SIZE / 2))
DECODE_D0_RANK_START=0
DECODE_D1_RANK_START=$DECODE_DP_SIZE_LOCAL
DECODE_START_PORT=9900
DECODE_LAST_PORT=$((DECODE_START_PORT + DECODE_DP_SIZE_LOCAL - 1))

: "${ROUND_DIR:?Set ROUND_DIR to a directory under /public-flash/liyan/glm52-vllm-ascend/runs}"
