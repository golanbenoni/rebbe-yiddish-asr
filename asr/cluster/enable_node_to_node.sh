#!/usr/bin/env bash
# Give the nodes SSH access to each other (LAN speed transfers) with a key generated on the coordinator.
#   cluster/enable_node_to_node.sh cluster/hosts.txt
set -u
HOSTS=${1:-cluster/hosts.txt}
hosts=(); while IFS= read -r h; do [[ -z "$h" || "$h" == \#* ]] && continue; hosts+=("$h"); done < "$HOSTS"
coord=${hosts[0]}
pub=$(ssh -n -o BatchMode=yes "$coord" '[ -f ~/.ssh/fleet_internal_ed25519 ] || ssh-keygen -q -t ed25519 -N "" -C "fleet-internal@$(hostname -s)" -f ~/.ssh/fleet_internal_ed25519; cat ~/.ssh/fleet_internal_ed25519.pub')
for h in "${hosts[@]}"; do
  ssh -n -o BatchMode=yes "$h" "mkdir -p ~/.ssh && chmod 700 ~/.ssh && grep -qF '$pub' ~/.ssh/authorized_keys 2>/dev/null || echo '$pub' >> ~/.ssh/authorized_keys; chmod 600 ~/.ssh/authorized_keys"
done
ssh -n -o BatchMode=yes "$coord" 'grep -q "fleet-internal" ~/.ssh/config 2>/dev/null || printf "\n# fleet-internal\nHost bigmac0? supermac0?\n  IdentityFile ~/.ssh/fleet_internal_ed25519\n  IdentitiesOnly yes\n  StrictHostKeyChecking accept-new\n" >> ~/.ssh/config'
echo "node-to-node key installed; test:"; ssh -n -o BatchMode=yes "$coord" 'for h in supermac02 bigmac01; do ssh -n -o BatchMode=yes -o ConnectTimeout=8 $h "echo ok from $(hostname -s) to \$(hostname -s)"; done'
