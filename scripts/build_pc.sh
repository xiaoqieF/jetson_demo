#!/usr/bin/env bash
set -euo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ros_distro="${ROS_DISTRO:-jazzy}"
source "/opt/ros/${ros_distro}/setup.bash"

cd "${workspace_dir}"
colcon build --symlink-install --packages-up-to argus_operator_ui \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3

echo "PC build complete. Run: source ${workspace_dir}/install/setup.bash"
