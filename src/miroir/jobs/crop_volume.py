"""A Block Storage volume for a bundle's whole crop vectors (`miroir.web.crops`, 164 GB).

The crops' whole vectors are read a few thousand rows at a time per search, from disk; they
do not fit beside the system, the job image, the weights and `pop-all` on a 100 GB root
volume. `add_crop_volume` makes a volume, attaches it to a machine, formats it if it is new
and mounts it where the bundle keeps them (`<web-index>/<bundle>/crops_whole`), listed in
`/etc/fstab` so it comes back at every boot. Used by the measuring job machine, and once by
`miroir demo crops-volume` on the demo machine.
"""

from __future__ import annotations

import logging
import shlex
from typing import TYPE_CHECKING

from miroir.web.crops import WHOLE_DIR

if TYPE_CHECKING:
    from miroir.jobs.scaleway import InstanceApi
    from miroir.jobs.scw_run import Shell

logger = logging.getLogger(__name__)

#: The whole vectors weigh 164 GB; room for them and a little more.
VOLUME_GB = 200
VOLUME_TAG = "miroir-crops"


def mount_point(web_index: str, bundle: str) -> str:
    return f"{web_index}/{bundle}/{WHOLE_DIR}"


def mount_script(volume_id: str, mount: str) -> str:
    """Finds the volume's disk (by its id, else the one disk with nothing on it), formats it
    when it holds no file system, and mounts it at `mount` now and at every boot."""
    vid, where = shlex.quote(volume_id), shlex.quote(mount)
    return f"""set -e
dev=""
for i in $(seq 60); do
  dev=$(ls -1 /dev/disk/by-id/ 2>/dev/null | grep -i -- {vid} | head -1 || true)
  if [ -n "$dev" ]; then dev=/dev/disk/by-id/$dev; break; fi
  free=$(lsblk -dnpo NAME,TYPE | awk '$2=="disk"{{print $1}}' | while read d; do
    if [ -z "$(lsblk -no MOUNTPOINT,FSTYPE "$d" | tr -d ' \\n')" ]; then echo "$d"; fi; done)
  if [ "$(echo "$free" | grep -c .)" = 1 ]; then dev=$free; break; fi
  sleep 2
done
[ -n "$dev" ] || {{ echo "no disk for volume {volume_id}"; exit 1; }}
blkid "$dev" > /dev/null || mkfs.ext4 -q -F "$dev"
uuid=$(blkid -s UUID -o value "$dev")
mkdir -p {where}
grep -q "$uuid" /etc/fstab || echo "UUID=$uuid {mount} ext4 defaults,nofail 0 2" >> /etc/fstab
mountpoint -q {where} || mount {where}
df -h {where} | tail -1
"""


def add_crop_volume(
    api: InstanceApi,
    shell: Shell,
    *,
    zone: str,
    server_id: str,
    mount: str,
    name: str,
    gb: int = VOLUME_GB,
    volume_id: str | None = None,
) -> str:
    """Make (unless `volume_id` is given), attach and mount the volume. Returns its id."""
    if volume_id is None:
        volume_id = api.create_block_volume(zone, name, gb, [VOLUME_TAG])
        logger.info("volume %s: %d GB made", volume_id, gb)
    api.attach_block_volume(zone, server_id, volume_id)
    out = shell.run(mount_script(volume_id, mount), timeout=600)
    logger.info("volume %s mounted at %s: %s", volume_id, mount, out.strip())
    return volume_id
