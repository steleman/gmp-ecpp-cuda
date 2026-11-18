# Setting up Slurm for `gmpecmfft-slurm` and `gmpecmfft-mpi`

Two setups: the development machine (one node, one GPU) and a
hypothetical multi-node GPU cluster. Every command marked `#` runs as
root.

**Nothing below has been run.** Slurm's daemons were never started on
the development machine, so this is a plan to follow and check step by
step, not a tested procedure. The facts about the development machine
in section 1 were checked; the commands and configuration were not.

---

## 1. The development machine: one node, one GPU

What is installed (checked):

| | |
|---|---|
| OS | Fedora 41, cgroup v2 |
| Slurm | 24.05.2 (`slurm`, `slurm-slurmctld`, `slurm-slurmd`, ...) |
| MUNGE | 0.5.16 (the `munge` user exists) |
| PMIx / Open MPI | PMIx 4.2.8; Open MPI 5.0.5 built against it; Slurm has `mpi_pmix` |
| GPU | one NVIDIA RTX PRO 4500 Blackwell, `/dev/nvidia0` |
| Slurm GPU plugins | `gres_gpu`, `gpu_generic` -- **no `gpu_nvml`**, so `AutoDetect=nvml` is not available and GPUs are listed by hand in `gres.conf` |
| `slurmd` unit | `Delegate=yes` (needed for cgroup v2) |
| Stock config | `/etc/slurm/slurm.conf` is a placeholder (`NodeName=localhost CPUs=1`); no `slurm` user; services disabled |

### 1.1 MUNGE

```sh
# ls -l /etc/munge/munge.key || sudo -u munge /usr/sbin/mungekey --verbose
# systemctl enable --now munge
$ munge -n | unmunge | grep STATUS          # STATUS: Success (0)
```

The key must be owned by `munge:munge`, mode 0600 (or 0400).

### 1.2 Describe the node

```sh
$ slurmd -C
```

prints a `NodeName=... CPUs=... RealMemory=...` line for this machine.
**Use its figures, not `nproc`'s.** On this machine `slurmd -C` reports
16 CPUs where `nproc` reports 32 (its CPU mixes core types, which
confuses the detection). Configuring more CPUs than `slurmd` detects
puts the node in DRAIN ("Low socket\*core\*thread count"). Lower
`RealMemory` a little below the reported figure, leaving room for the
system.

### 1.3 `/etc/slurm/slurm.conf`

Keep a copy of the stock file, then replace it. `HOST` is the short
hostname (`hostname -s`); the `NodeName` line is `slurmd -C`'s, with
`RealMemory` lowered and `Gres` and `State` added.

```
ClusterName=gmpecm
SlurmctldHost=HOST
SlurmUser=root                 # Fedora's packaged default; fine on one machine
AuthType=auth/munge
StateSaveLocation=/var/spool/slurm/ctld
SlurmdSpoolDir=/var/spool/slurm/d
SlurmctldPidFile=/var/run/slurm/slurmctld.pid
SlurmdPidFile=/var/run/slurm/slurmd.pid
SlurmctldLogFile=/var/log/slurm/slurmctld.log
SlurmdLogFile=/var/log/slurm/slurmd.log
MpiDefault=pmix
ProctrackType=proctrack/cgroup
TaskPlugin=task/cgroup,task/affinity
SelectType=select/cons_tres     # share a node between jobs; one GPU each
SelectTypeParameters=CR_Core_Memory
GresTypes=gpu
SchedulerType=sched/backfill
ReturnToService=2
NodeName=HOST CPUs=16 Boards=1 SocketsPerBoard=1 CoresPerSocket=8 ThreadsPerCore=2 RealMemory=120000 Gres=gpu:1 State=UNKNOWN
PartitionName=gpu Nodes=ALL Default=YES MaxTime=INFINITE State=UP
```

### 1.4 `/etc/slurm/gres.conf`

```
NodeName=HOST Name=gpu File=/dev/nvidia0
```

The stock `/etc/slurm/cgroup.conf` (`ConstrainCores`, `ConstrainDevices`,
`ConstrainRAMSpace`, `ConstrainSwapSpace`, all `yes`) can stay.
`ConstrainDevices` means a job sees only the GPU it was given.

### 1.5 Start and check

```sh
# systemctl enable --now slurmctld slurmd
$ sinfo                                   # partition gpu, node idle
$ scontrol show node HOST | grep -E 'State|Gres|Reason'
$ srun --gres=gpu:1 nvidia-smi -L         # the RTX PRO 4500
```

If the node is DOWN, DRAIN or INVALID, `scontrol show node` gives the
reason, and the details are in `/var/log/slurm/slurmd.log` and
`slurmctld.log`. After fixing the configuration:

```sh
# scontrol reconfigure
# scontrol update nodename=HOST state=resume
```

If `slurmd` will not start with cgroup errors in its log, a single-user
test machine can do without cgroups: `ProctrackType=proctrack/linuxproc`,
`TaskPlugin=task/affinity`. GPU isolation then rests on
`CUDA_VISIBLE_DEVICES` alone, which Slurm still sets.

### 1.6 Try `gmpecmfft-slurm`

The same case `make check-fft` uses (the 2000-bit number from
`bench/fft-numbers.txt`, factor 170286825566383, found in batch 10 at
these bounds):

```sh
$ awk '$1==2000{print $3}' bench/fft-numbers.txt |
    tools/gmpecmfft-slurm submit -d /tmp/r1 -n 1 -c 32 -1 2000 -2 0
$ squeue                                  # the array, until it finishes
$ tools/gmpecmfft-slurm status /tmp/r1    # factor ... (task 0 batch 10 seed 11 ...)
$ cat /tmp/r1/task-0.out
```

On one machine the run directory can be on local disk. With one GPU,
the array's tasks run one after another. A task without `-b` runs until
a factor turns up, so give `-n 1`, or `-b K` so that each task finishes
and the next starts.

### 1.7 `gmpecmfft-mpi` under Slurm

The coordinator plus two workers sharing the one GPU, as `make
check-fft` runs it with `mpirun` (the number goes to rank 0 on stdin):

```sh
$ sbatch -n 3 --gres=gpu:1 -o /tmp/mpi-%j.out --wrap \
    "srun --mpi=pmix ./gmpecmfft-mpi -c 32 -1 2000 -2 0 -v < /tmp/n2000.txt"
```

where `/tmp/n2000.txt` holds the number. Whether all three tasks of the
step see the job's one GPU, rather than only the first, is one of the
things to check here.

---

## 2. A multi-node cluster (hypothetical)

The example layout: a head node `head` (runs `slurmctld`, users log in
there) and four compute nodes `gpu01` .. `gpu04`, each with four GPUs
**of the same model**. Both tools assume one GPU type: with a mix, the
faster GPUs finish their share and then sit idle at the end
(`gmpecmfft-slurm` divides the work in advance, so it suffers from a mix
even more than `gmpecmfft-mpi`).

### 2.1 Every node

- The same OS, the same Slurm version and the same MUNGE.
- On compute nodes, an NVIDIA driver that supports CUDA 13.2. Install
  the CUDA 13.2 runtime and cuFFT under the same path as on the build
  machine: `libgmpecmfft` finds cuFFT through the rpath
  `/usr/local/cuda-13.2/lib64`.
- GMP. For `gmpecmfft-mpi`, also Open MPI and PMIx.
- Name resolution for all nodes (DNS, or the same `/etc/hosts`).
- Time synchronization (`chronyd`). A MUNGE credential is refused after
  about five minutes of clock skew.
- The same UIDs and GIDs everywhere, for `munge`, `slurm` and every user
  who submits jobs (LDAP/FreeIPA, or identical `passwd` and `group`
  files).
- A shared filesystem mounted at the same path on every node, for
  example `/shared` over NFS. It holds:
  - the gmp-ecpp build directory (`gmpecmfft`, `gmpecmfft-mpi`,
    `libgmpecmfft.so*`, `libgmpecpp.so*`, `tools/`). The binaries find
    their libraries through `$ORIGIN`, so the directory moves as a
    unit.
  - the run directories of `gmpecmfft-slurm`, which must be on a
    filesystem every node sees.

### 2.2 MUNGE: one key for the whole cluster

```sh
head# sudo -u munge /usr/sbin/mungekey --verbose
head# for h in gpu01 gpu02 gpu03 gpu04; do
        scp -p /etc/munge/munge.key $h:/etc/munge/munge.key
        ssh $h 'chown munge:munge /etc/munge/munge.key; chmod 0600 /etc/munge/munge.key'
      done
all#  systemctl enable --now munge
head$ munge -n | ssh gpu01 unmunge | grep STATUS    # Success, across nodes
```

### 2.3 The `slurm` user and directories

On a cluster the controller should not run as root:

```sh
all#  useradd -r -u 980 -U -s /sbin/nologin -d /var/spool/slurm slurm   # same UID everywhere
head# mkdir -p /var/spool/slurm/ctld /var/log/slurm
head# chown slurm:slurm /var/spool/slurm/ctld /var/log/slurm
node# mkdir -p /var/spool/slurm/d /var/log/slurm   # slurmd itself runs as root
```

### 2.4 `/etc/slurm/slurm.conf`: identical on every node

```
ClusterName=gmpecm
SlurmctldHost=head
SlurmUser=slurm
AuthType=auth/munge
StateSaveLocation=/var/spool/slurm/ctld
SlurmdSpoolDir=/var/spool/slurm/d
SlurmctldPidFile=/var/run/slurm/slurmctld.pid
SlurmdPidFile=/var/run/slurm/slurmd.pid
SlurmctldLogFile=/var/log/slurm/slurmctld.log
SlurmdLogFile=/var/log/slurm/slurmd.log
MpiDefault=pmix
ProctrackType=proctrack/cgroup
TaskPlugin=task/cgroup,task/affinity
SelectType=select/cons_tres
SelectTypeParameters=CR_Core_Memory
GresTypes=gpu
SchedulerType=sched/backfill
ReturnToService=2
SrunPortRange=60001-63000
# From `slurmd -C` on a compute node; RealMemory a little lower.
NodeName=gpu[01-04] CPUs=64 RealMemory=500000 Gres=gpu:4 State=UNKNOWN
PartitionName=gpu Nodes=gpu[01-04] Default=YES MaxTime=INFINITE State=UP
```

(The `CPUs` and `RealMemory` figures are placeholders.) Instead of
copying the file around, Slurm's *configless* mode can serve it: add
`SlurmctldParameters=enable_configless` on the head node, and start
each node's `slurmd` with `--conf-server head`, set through
`SLURMD_OPTIONS` in `/etc/sysconfig/slurmd`. `gres.conf` is then served
too.

### 2.5 `/etc/slurm/gres.conf`

```
NodeName=gpu[01-04] Name=gpu File=/dev/nvidia[0-3]
```

If your Slurm package was built with NVML (it has
`/usr/lib64/slurm/gpu_nvml.so`; Fedora's does not), `AutoDetect=nvml`
replaces the list.

### 2.6 Firewall

```sh
head# firewall-cmd --permanent --add-port=6817/tcp            # slurmctld
node# firewall-cmd --permanent --add-port=6818/tcp            # slurmd
node# firewall-cmd --permanent --add-port=60001-63000/tcp     # srun (SrunPortRange)
all#  firewall-cmd --reload
```

Open MPI between nodes uses further, dynamic TCP ports. On a private
cluster network it is simpler to trust that network on every node:

```sh
all#  firewall-cmd --permanent --zone=trusted --add-source=10.0.0.0/24 && firewall-cmd --reload
```

### 2.7 Start and check

```sh
head# systemctl enable --now slurmctld
node# systemctl enable --now slurmd
head$ sinfo -N -l                           # four nodes, idle, gpu:4 each
head$ srun -N 4 hostname                    # one line per node
head$ srun -N 1 --gres=gpu:4 nvidia-smi -L  # four GPUs, one model
```

### 2.8 Running

**`gmpecmfft-slurm`**: one array task per GPU, so 16 tasks.
`cons_tres` puts four on each node, one GPU each.

```sh
head$ echo N | /shared/gmp-ecpp/tools/gmpecmfft-slurm submit -d /shared/runs/r1 \
        -n 16 -1 1e6 -2 5e7 -c 256 -- --partition=gpu --time=48:00:00
head$ squeue -r                              # the array's tasks
head$ /shared/gmp-ecpp/tools/gmpecmfft-slurm status /shared/runs/r1
```

Give `-c` explicitly. A batch is then the same on every GPU, and
`status`'s single-GPU command reproduces the batch that found the
factor.

**`gmpecmfft-mpi`**: one rank per GPU, plus the coordinator: 16 + 1 =
17 tasks on four nodes. Pass the number on stdin (it goes to rank 0).
Numbers of hundreds of thousands of digits are too long for the
command line, whose single-argument limit is 128 KiB.

```sh
head$ sbatch -N 4 -n 17 --gres=gpu:4 -p gpu -o /shared/runs/mpi-%j.out --wrap \
        "srun --mpi=pmix /shared/gmp-ecpp/gmpecmfft-mpi -1 1e6 -2 5e7 -c 256 -v < /shared/runs/n.txt"
```

Slurm's default block distribution should put ranks 0-4 (the
coordinator and four workers) on `gpu01` and four ranks on each other
node; check with `srun -N 4 -n 17 hostname | sort | uniq -c`. Each
worker takes the GPU given by its index among the workers on its node,
modulo the node's GPU count. If a node gets more workers than GPUs, the
tool warns.
