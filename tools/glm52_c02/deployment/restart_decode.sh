#!/usr/bin/env bash
# Run on the corresponding D host, after an idle-service check.
set -euo pipefail
role=${1:?usage: restart_decode.sh d0|d1 RUN_ID}
run_id=${2:?usage: restart_decode.sh d0|d1 RUN_ID}
[[ "$role" == d0 || "$role" == d1 ]]
[[ "$run_id" =~ ^[A-Za-z0-9._-]+$ ]]
root=/public-flash/liyan/glm52-vllm-ascend
experiment=$root/experiments/glm53-migration/C02
container=glm52-vllm-$role
round=$root/runs/$run_id
mkdir -p "$round/config" "$round/log"

docker exec -e C02_ROLE="$role" -e C02_ROUND="$round" "$container" bash -lc '
set -euo pipefail
cd /vllm-workspace/vllm-ascend
experiment=/public-flash/liyan/glm52-vllm-ascend/experiments/glm53-migration/C02
git diff > "$C02_ROUND/config/runtime-before-$C02_ROLE.patch"
if git apply --reverse --check "$experiment/runtime.patch" 2>/dev/null; then
    echo "C02 is already applied"
else
    git apply --check "$experiment/runtime.patch"
    git apply "$experiment/runtime.patch"
fi
python3 - "$experiment/vllm-ascend" <<PY
import hashlib
import sys
from pathlib import Path
reference = Path(sys.argv[1])
for relative in ("vllm_ascend/ascend_config.py", "vllm_ascend/attention/sfa_v1.py"):
    expected = (reference / relative).read_bytes()
    if relative.endswith("sfa_v1.py"):
        old = b"            or qt is None\n"
        assert expected.count(old) == 1
        expected = expected.replace(old, b"")
    actual = Path(relative).read_bytes()
    assert actual == expected, f"Unexpected runtime difference: {relative}"
    print(relative, hashlib.sha256(actual).hexdigest())
PY
git diff > "$C02_ROUND/config/runtime-after-$C02_ROLE.patch"
cp "$experiment/runtime.patch" "$C02_ROUND/config/c02-runtime.patch"
cp "$experiment/scripts/"* "$C02_ROUND/config/"
' > "$round/config/apply-$role.log" 2>&1

docker restart "$container" > "$round/config/restart-$role.log"
docker exec \
    -e PERF_MODE=optimized -e DECODE_SCENE=dp2tp16 \
    -e MTP_MODEL_PATH="$root/models/GLM-5.2-w8a8-mtp-rot" -e MTP_VIEW_MODE=loader \
    -e OPTIMIZED_P_SFA_C8=false -e OPTIMIZED_D_SFA_C8=false \
    -e SFA_UNQUANTIZED_NATIVE_PATCH=true -e ITL_EVENT_RECORDER=true \
    -e ROUND_DIR="$round" \
    "$container" bash "$experiment/scripts/start_role.sh" "$role" \
    > "$round/config/start-$role.log" 2>&1
