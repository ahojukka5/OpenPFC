<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# LUMI-G topology observability for issue #106

Separate three classes of statement. Do not promote (2) or (3) into
(1).

## 1. Documented physical topology

CSC's [network page](https://docs.lumi-supercomputer.eu/hardware/network/)
and the LUMI architecture talks state:

- HPE Cray Slingshot-11 dragonfly;
- LUMI-G: 24 electrical groups of 124 nodes, last group 126;
- 32 Rosetta switches per group, 16 endpoints per switch;
- four Slingshot endpoints per LUMI-G node (one per MI250x), each on a
  different switch;
- intra-group copper, inter-group optical.

Those sizes are **hardware documentation**. They are not a per-job
label. 124, 248, and 496 nodes are candidate sampling points because
they are integer multiples of the documented group size. They are not
verified group-aligned allocations.

Slurm training materials mention node features of the form `xNNNN` that
correspond to Slingshot interconnect groups and can be used as a
constraint. That is a scheduler feature list, not automatic job
metadata.

## 2. Metadata this campaign actually records

The #106 batch script writes `placement/` inside each allocation:

| source | field | observed on 136-node job 22186611 |
|--------|--------|-------------------------------------|
| `scontrol show hostnames` | `nid00xxxx` hostnames | 136 names, nid range 5234–7782 |
| `/proc/cray_xt/nid` | numeric nid | **not present** (`nid=na`) |
| `/proc/cray_xt/cname`, `/etc/xname` | Cray xname | **not present** (`xname=na`) |
| `/sys/class/cxi/cxi*` | Cassini NICs | `cxi0,cxi1,cxi2,cxi3` |
| Slurm | job id, nodelist, partition, account | present |
| rank map | `SLURM_PROCID` / host / localid | present |

Cabinet fields parsed from an xname (`x<cabinet>c…`) are therefore empty
on the completed allocation. They are not electrical-group identifiers
even when present.

Pre-execution, without an allocation, the only topology-class facts that
are determined are:

- 8 GCDs per node;
- whether a reshape's peer group is entirely on-node (`nodes == 1`) or
  necessarily off-node (`nodes > 1` for an all-rank reshape).

## 3. Inferences that must not be treated as fact

- Host nid numerical proximity is **not** a group id.
- `nodes % 124 == 0` is **not** evidence that the job sat inside one
  electrical group.
- Parsed xname cabinets, if they later appear, are **not** Slingshot
  group labels.
- CXI device names show four NICs exist; they do not name the switch or
  the group.

If a later allocation exposes `scontrol show node` features containing
`xNNNN`, record the feature string as *observed scheduler metadata* and
still do not equate it with the documented 124-node group without a
join table that CSC does not provide inside the job.
