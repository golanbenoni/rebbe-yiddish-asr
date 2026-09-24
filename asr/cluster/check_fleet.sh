#!/usr/bin/env bash
# Test key-based SSH to every host and print its specs.   cluster/check_fleet.sh cluster/hosts.txt
HOSTS=${1:?hosts file}; ok=0; n=0
while IFS= read -r h; do [[ -z "$h" || "$h" == \#* ]] && continue; n=$((n+1))
  out=$(ssh -n -o BatchMode=yes -o ConnectTimeout=8 "$h" 'echo "$(hostname -s) | $(sysctl -n machdep.cpu.brand_string) | $(( $(sysctl -n hw.memsize)/1073741824 )) GB | $(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || sysctl -n hw.ncpu) P-cores | macOS $(sw_vers -productVersion) | python3: $(command -v python3 >/dev/null && python3 --version 2>&1 || echo none) | free: $(df -h ~ | tail -1 | awk "{print \$4}")"' 2>&1) && { ok=$((ok+1)); echo "OK   $out"; } || echo "FAIL $h: $(echo "$out" | tail -1)"
done < "$HOSTS"
echo "$ok/$n hosts reachable with the fleet key"
