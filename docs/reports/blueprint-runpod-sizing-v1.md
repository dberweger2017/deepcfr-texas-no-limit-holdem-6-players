# RunPod blueprint sizing check

## First pod and hardware shown by RunPod

On 2026-09-24 the signed-in RunPod console deployed pod `k9rdph2fwhym87` (`blueprint-learning-check-20260924`) from the `runpod/base:0.7.0-ubuntu2004` template in region **US-CA-2**. The selected tier was **3 GHz memory-optimized, 8 vCPU, 64 GB RAM**, with a **20 GB disposable container disk** and no network volume. The console quoted **$0.44/hour CPU** and **$0.003/hour disk**. Its pod details identified the underlying processor as **AMD EPYC 7713 64-Core Processor**. That host model does not mean all 64 physical cores are assigned to this pod; the allocation shown was 8 vCPU.

The account's imported SSH key did not match any available private key on the local Mac or M4, so no commands or checkpoint transfer ran inside this pod. It was stopped with the console showing **$0.00/hour** continuing cost. The displayed balance moved from **$18.07 to $18.05**. No scientific result or full topology measurement came from this first attempt. Adding this Mac's public key to the account is pending the required security-access confirmation; the pod will be started again only after access is resolved.

The completed run will add raw `lscpu`, `free`, cgroup quota, storage and filesystem records, plus measured per-worker speed and memory. These are needed to compare this 8-vCPU allocation with a later 128–512 GB host; the RunPod tier name and processor label alone do not establish available cores, memory bandwidth, or scaling.
