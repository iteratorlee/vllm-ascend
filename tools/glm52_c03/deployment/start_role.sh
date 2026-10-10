#!/usr/bin/env bash
set -euo pipefail

role=${1:?usage: start_role.sh p0|p1|d0|d1}
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./config.sh
export GLM52_GLOO_TIMEOUT_SECONDS=${GLM52_GLOO_TIMEOUT_SECONDS:-7200}
export PYTHONPATH="/public-flash/liyan/glm52-vllm-ascend/patches${PYTHONPATH:+:$PYTHONPATH}"
SFA_UNQUANTIZED_NATIVE_PATCH=${SFA_UNQUANTIZED_NATIVE_PATCH:-false}
ITL_EVENT_RECORDER=${ITL_EVENT_RECORDER:-false}
[[ "$ITL_EVENT_RECORDER" == true || "$ITL_EVENT_RECORDER" == false ]] || {
    echo "ITL_EVENT_RECORDER must be true or false" >&2
    exit 2
}
[[ "$SFA_UNQUANTIZED_NATIVE_PATCH" == true || "$SFA_UNQUANTIZED_NATIVE_PATCH" == false ]] || {
    echo "SFA_UNQUANTIZED_NATIVE_PATCH must be true or false" >&2
    exit 2
}
if [[ "$SFA_UNQUANTIZED_NATIVE_PATCH" == true ]]; then
    [[ "$PERF_MODE" == optimized && "$OPTIMIZED_D_SFA_C8" == false && "$OPTIMIZED_P_SFA_C8" == false ]] || {
        echo "SFA unquantized native patch requires optimized P/D sparse SFA C8=false" >&2
        exit 2
    }
fi
if [[ "$MTP_VIEW_MODE" == loader ]]; then
    [[ -d "$MTP_MODEL_PATH" ]] || {
        echo "MTP loader view directory missing: $MTP_MODEL_PATH" >&2
        exit 2
    }
    export GLM52_MTP_VIEW_PATH="$MTP_MODEL_PATH"
fi
mkdir -p "$ROUND_DIR/log/ascend" "$ROUND_DIR/bench" "$ROUND_DIR/config"
cp ./config.sh "$ROUND_DIR/config/config-$role.sh"
if [[ "$role" == p0 ]]; then
    cp ./config.sh "$ROUND_DIR/config/config.sh"
fi
cat > "$ROUND_DIR/config/$role.env" <<EOF
timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
hostname=$(hostname)
role=$role
MODEL_PATH=$MODEL_PATH
PERF_MODE=$PERF_MODE
DECODE_SCENE=$DECODE_SCENE
DECODE_DP_SIZE=$DECODE_DP_SIZE
DECODE_TP_SIZE=$DECODE_TP_SIZE
ROUND_DIR=$ROUND_DIR
NIC_NAME=$NIC_NAME
GLM52_GLOO_TIMEOUT_SECONDS=$GLM52_GLOO_TIMEOUT_SECONDS
MTP_MODEL_PATH=${MTP_MODEL_PATH:-}
MTP_VIEW_MODE=$MTP_VIEW_MODE
PREFILL_KV_PORT=$PREFILL_KV_PORT
OPTIMIZED_D_SFA_C8=$OPTIMIZED_D_SFA_C8
OPTIMIZED_P_SFA_C8=$OPTIMIZED_P_SFA_C8
SFA_UNQUANTIZED_NATIVE_PATCH=$SFA_UNQUANTIZED_NATIVE_PATCH
ITL_EVENT_RECORDER=$ITL_EVENT_RECORDER
EOF

if [[ "$ITL_EVENT_RECORDER" == true && ( "$role" == d0 || "$role" == d1 ) ]]; then
    export GLM52_ITL_OUTPUT_DIR="$ROUND_DIR/bench/server-events"
    export GLM52_ITL_ROLE="$role"
    cp /public-flash/liyan/glm52-vllm-ascend/patches/patch_itl_recorder.py \
       /public-flash/liyan/glm52-vllm-ascend/patches/glm52_itl_recorder.py \
       "$ROUND_DIR/config/"
    python3 /public-flash/liyan/glm52-vllm-ascend/patches/patch_itl_recorder.py \
        > "$ROUND_DIR/config/itl-recorder-patch-$role.json"
fi

if [[ "$MTP_VIEW_MODE" == loader ]]; then
    if [[ "$role" == p0 || "$role" == d0 ]]; then
        cp /public-flash/liyan/glm52-vllm-ascend/patches/patch_mtp_loader_view.py \
            "$ROUND_DIR/config/patch_mtp_loader_view-$role.py"
    fi
    python3 /public-flash/liyan/glm52-vllm-ascend/patches/patch_mtp_loader_view.py \
        > "$ROUND_DIR/config/mtp-loader-patch-$role.txt"
fi

if [[ "$SFA_UNQUANTIZED_NATIVE_PATCH" == true && ( "$role" == d0 || "$role" == d1 ) ]]; then
    cp /public-flash/liyan/glm52-vllm-ascend/patches/patch_sfa_unquantized.py \
        "$ROUND_DIR/config/patch_sfa_unquantized-$role.py"
    python3 /public-flash/liyan/glm52-vllm-ascend/patches/patch_sfa_unquantized.py \
        > "$ROUND_DIR/config/sfa-patch-$role.txt"
fi

if [[ "$role" == p0 || "$role" == p1 ]]; then
    if [[ "$role" == p0 ]]; then
        kv_ip=$P0_IP
        kv_rank_offset=0
    else
        kv_ip=$P1_IP
        kv_rank_offset=16
    fi
    python3 - "$kv_ip" "$PREFILL_KV_PORT" "$kv_rank_offset" <<'PY'
import socket
import sys

ip, base, offset = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
for port in range(base + offset, base + offset + 16):
    with socket.socket() as sock:
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((ip, port))
            sock.listen(1)
        except OSError as exc:
            raise SystemExit(f"Mooncake KV port {ip}:{port} is unavailable: {exc}")
print(f"Mooncake KV ports {ip}:{base + offset}-{base + offset + 15} are available")
PY
fi
if [[ "$role" == p0 ]]; then
    cp ./prefill-*.sh ./decode-*.sh ./proxy.sh ./start_role.sh "$ROUND_DIR/config/"
    sha256sum ./config.sh ./prefill-*.sh ./decode-*.sh ./proxy.sh \
        ./launch_online_dp.py ./load_balance_proxy_server_example.py \
        > "$ROUND_DIR/config/scripts.sha256"
fi

case "$role" in
    p0) bash ./prefill-0.sh ;;
    p1) bash ./prefill-1.sh ;;
    d0)
        python3 ./launch_online_dp.py \
            --template ./decode-0.sh \
            --dp-size "$DECODE_DP_SIZE" --tp-size "$DECODE_TP_SIZE" \
            --dp-size-local "$DECODE_DP_SIZE_LOCAL" \
            --dp-rank-start "$DECODE_D0_RANK_START" \
            --dp-address "$D0_IP" --dp-rpc-port 12321 \
            --vllm-start-port "$DECODE_START_PORT"
        ;;
    d1)
        python3 ./launch_online_dp.py \
            --template ./decode-1.sh \
            --dp-size "$DECODE_DP_SIZE" --tp-size "$DECODE_TP_SIZE" \
            --dp-size-local "$DECODE_DP_SIZE_LOCAL" \
            --dp-rank-start "$DECODE_D1_RANK_START" \
            --dp-address "$D0_IP" --dp-rpc-port 12321 \
            --vllm-start-port "$DECODE_START_PORT"
        ;;
    *) echo "invalid role: $role" >&2; exit 2 ;;
esac
