#!/usr/bin/env bash
# Run once per corresponding host, after the shared idle-service check.
set -euo pipefail
role=${1:?usage: restart_role.sh p0|p1|d0|d1 PROFILE_DIRECTORY}
profile=${2:?usage: restart_role.sh p0|p1|d0|d1 PROFILE_DIRECTORY}
[[ "$role" =~ ^[pd][01]$ ]]
[[ "$profile" == /public-flash/liyan/glm52-vllm-ascend/profiling/* ]]
root=/public-flash/liyan/glm52-vllm-ascend
container=glm52-vllm-$role
mkdir -p "$profile/config" "$profile/log"

docker exec -i "$container" python3 - "$root" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for name in ("20261010-c03-fused-b0", "20261010-c03-fused-b1"):
    bench = root / "runs" / name / "bench"
    result = json.loads((bench / "results.json").read_text())
    assert result["summary"]["completed"] == 64 and result["summary"]["failed"] == 0
    assert (bench / "matched-server-events/manifest.json").exists()
PY

docker exec "$container" bash -lc '
cd /vllm-workspace/vllm-ascend
git diff
' > "$profile/config/runtime-before-$role.patch"
docker restart "$container" > "$profile/config/restart-$role.log"
docker exec \
    -e PERF_MODE=optimized -e DECODE_SCENE=dp2tp16 \
    -e MTP_MODEL_PATH="$root/models/GLM-5.2-w8a8-mtp-rot" -e MTP_VIEW_MODE=loader \
    -e OPTIMIZED_P_SFA_C8=false -e OPTIMIZED_D_SFA_C8=false \
    -e SFA_UNQUANTIZED_NATIVE_PATCH=true -e ITL_EVENT_RECORDER=true \
    -e ROUND_DIR="$profile" \
    "$container" bash "$profile/scripts/start_role.sh" "$role" \
    > "$profile/config/start-$role.log" 2>&1
if [[ "$role" == p0 ]]; then
    docker exec \
        -e PERF_MODE=optimized -e DECODE_SCENE=dp2tp16 \
        -e OPTIMIZED_P_SFA_C8=false -e OPTIMIZED_D_SFA_C8=false \
        -e ROUND_DIR="$profile" \
        "$container" bash -lc 'cd "$ROUND_DIR/scripts"; bash ./proxy.sh' \
        > "$profile/config/start-proxy.log" 2>&1
fi
