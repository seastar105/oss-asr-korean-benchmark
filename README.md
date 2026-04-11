# Korean OSS ASR Benchmark

As of March 2026, several speech recognition models officially support Korean. However, from a user’s perspective, the most important question is still: `Is it better than Whisper?`

The problem is that most companies either report results only on datasets such as Common Voice or FLEURS, or do not report Korean performance at all. To make comparison easier, I ran my own evaluation using publicly available AI Hub datasets.

## Update

2026.04.11: Add FunAudioLLM/SenseVoiceSmall, and RaonSpeech/Raon-Speech-9B. Change text normalization method.

## Results (CER, RTF)

**NOTE: Since all test utterances are short (30 seconds or less), these results should not be interpreted as long-form ASR performance.**

|                  Model                  |  **Average**  |  CV 15 |  FLEURS  |  Callcenter  |  Conference  |  Callcenter2 |  Lecture  |  Kspon Clean  |  Kspon Other  |  RTF   |
|-----------------------------------------|-----------|---------------------|----------|--------------|--------------|-------------------------|-----------|--------------------------|--------------------------|--------|
| mistralai/Voxtral-Mini-4B-Realtime-2602 |    22.25% |              10.61% |    4.50% |       46.12% |       18.63% |                  17.17% |    18.60% |                   36.04% |                   26.29% | 0.7647 |
|                     Qwen/Qwen3-ASR-0.6B |    17.29% |               8.30% |    4.26% |       59.62% |       13.78% |                   8.49% |    12.79% |                   17.62% |                   13.47% | 0.1936 |
|                   openai/whisper-medium |    16.53% |               7.90% |    4.37% |       30.53% |       21.78% |                  15.18% |    14.20% |                   20.29% |                   18.03% | 0.3233 |
|    CohereLabs/cohere-transcribe-03-2026 |    15.97% |               6.21% |    4.91% |       27.64% |       16.32% |                   8.69% |    16.24% |                   27.50% |                   20.27% | 0.2458 |
|             FunAudioLLM/SenseVoiceSmall |    12.72% |               7.81% |    5.70% |       19.92% |       16.23% |                   9.36% |    13.90% |                   14.79% |                   14.04% | 0.003* |
|                     Qwen/Qwen3-ASR-1.7B |     9.78% |               5.73% |    2.96% |       15.87% |       11.56% |                   6.37% |    10.00% |                   14.96% |                   10.82% |  0.191 |
|           openai/whisper-large-v3-turbo |     9.36% |               5.48% |    3.22% |       10.00% |       10.14% |                   4.17% |     9.59% |                   17.96% |                   14.31% | 0.2302 |
|                 openai/whisper-large-v3 |     7.83% |               5.83% |    3.01% |        5.34% |        9.41% |                   3.74% |     8.07% |                   14.16% |                   13.08% |  0.298 |
|       seastar105/whisper-medium-komixv2 |     7.05% |               6.44% |    4.34% |        5.78% |        9.33% |                   4.90% |     8.37% |                    8.91% |                    8.31% | 0.2484 |
|               RaonSpeech/Raon-Speech-9B |     6.44% |               4.43% |    *2.89% |        5.72% |        8.98% |                   3.48% |     8.27% |                    *8.66% |                    *9.10% | 0.7184 |

RTX Pro 6000 Blackwell was used for Raon-Speech's server, since i could not run server on rtx 5090 machine.

RTF for sensevoice was measured in batched inference.

**NOTE: CER values for FLEURS, Ksponspeech are different from their technical report. This difference may come from different reference text.**

`whisper-large-v3` still delivers the best average performance overall, and the gap is especially large on the `Callcenter` dataset.

`whisper-medium-komixv2` is a Korean fine-tuned model, and it shows that fine-tuning is still highly effective when applying ASR models to a specific domain.

This effect is particularly large on `Callcenter`, whose speech domain differs significantly from the others, and on `KsponSpeech`, whose text domain is also quite different. This suggests that, from an operational cost perspective, it may be more efficient to take a lower-RTF model and fine-tune it on domain-specific data.

## Environment

All evaluations were conducted by serving each model with vLLM (`vllm serve <model-name>`) and then sending requests from 32 threads to collect ASR outputs. Please refer to [run_asr.py](./run_asr.py) and [run_asr_stream.py](./run_asr_stream.py).

The evaluation server specifications are shown below.

<details>

<summary>Environment</summary>

```text
==============================
        System Info
==============================
OS                           : Ubuntu 24.04.3 LTS (x86_64)
GCC version                  : (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0
Clang version                : Could not collect
CMake version                : version 3.28.3
Libc version                 : glibc-2.39

==============================
       PyTorch Info
==============================
PyTorch version              : 2.10.0+cu128
Is debug build               : False
CUDA used to build PyTorch   : 12.8
ROCM used to build PyTorch   : N/A

==============================
      Python Environment
==============================
Python version               : 3.12.3 (main, Aug 14 2025, 17:47:21) [GCC 13.3.0] (64-bit runtime)
Python platform              : Linux-6.8.0-90-generic-x86_64-with-glibc2.39

==============================
       CUDA / GPU Info
==============================
Is CUDA available            : True
CUDA runtime version         : 12.8.93
CUDA_MODULE_LOADING set to   :
GPU models and configuration : GPU 0: NVIDIA GeForce RTX 5090
Nvidia driver version        : 590.48.01
cuDNN version                : Probably one of the following:
/usr/lib/x86_64-linux-gnu/libcudnn.so.9.8.0
/usr/lib/x86_64-linux-gnu/libcudnn_adv.so.9.8.0
/usr/lib/x86_64-linux-gnu/libcudnn_cnn.so.9.8.0
/usr/lib/x86_64-linux-gnu/libcudnn_engines_precompiled.so.9.8.0
/usr/lib/x86_64-linux-gnu/libcudnn_engines_runtime_compiled.so.9.8.0
/usr/lib/x86_64-linux-gnu/libcudnn_graph.so.9.8.0
/usr/lib/x86_64-linux-gnu/libcudnn_heuristic.so.9.8.0
/usr/lib/x86_64-linux-gnu/libcudnn_ops.so.9.8.0
HIP runtime version          : N/A
MIOpen runtime version       : N/A
Is XNNPACK available         : True

==============================
          CPU Info
==============================
Architecture:                         x86_64
CPU op-mode(s):                       32-bit, 64-bit
Address sizes:                        52 bits physical, 57 bits virtual
Byte Order:                           Little Endian
CPU(s):                               128
On-line CPU(s) list:                  0-127
Vendor ID:                            GenuineIntel
Model name:                           INTEL(R) XEON(R) GOLD 6530
CPU family:                           6
Model:                                207
Thread(s) per core:                   2
Core(s) per socket:                   32
Socket(s):                            2
Stepping:                             2
CPU(s) scaling MHz:                   22%
CPU max MHz:                          4000.0000
CPU min MHz:                          800.0000
BogoMIPS:                             4200.00
Flags:                                fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf tsc_known_freq pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb cat_l3 cat_l2 cdp_l3 intel_ppin cdp_l2 ssbd mba ibrs ibpb stibp ibrs_enhanced tpr_shadow flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb intel_pt avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local split_lock_detect user_shstk avx_vnni avx512_bf16 wbnoinvd dtherm ida arat pln pts hwp hwp_act_window hwp_epp hwp_pkg_req vnmi avx512vbmi umip pku ospke waitpkg avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg tme avx512_vpopcntdq la57 rdpid bus_lock_detect cldemote movdiri movdir64b enqcmd fsrm md_clear serialize tsxldtrk pconfig arch_lbr ibt amx_bf16 avx512_fp16 amx_tile amx_int8 flush_l1d arch_capabilities ibpb_exit_to_user
Virtualization:                       VT-x
L1d cache:                            3 MiB (64 instances)
L1i cache:                            2 MiB (64 instances)
L2 cache:                             128 MiB (64 instances)
L3 cache:                             320 MiB (2 instances)
NUMA node(s):                         2
NUMA node0 CPU(s):                    0-31,64-95
NUMA node1 CPU(s):                    32-63,96-127
Vulnerability Gather data sampling:   Not affected
Vulnerability Itlb multihit:          Not affected
Vulnerability L1tf:                   Not affected
Vulnerability Mds:                    Not affected
Vulnerability Meltdown:               Not affected
Vulnerability Mmio stale data:        Not affected
Vulnerability Reg file data sampling: Not affected
Vulnerability Retbleed:               Not affected
Vulnerability Spec rstack overflow:   Not affected
Vulnerability Spec store bypass:      Mitigation; Speculative Store Bypass disabled via prctl
Vulnerability Spectre v1:             Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:             Mitigation; Enhanced / Automatic IBRS; IBPB conditional; RSB filling; PBRSB-eIBRS SW sequence; BHI BHI_DIS_S
Vulnerability Srbds:                  Not affected
Vulnerability Tsx async abort:        Not affected
Vulnerability Vmscape:                Mitigation; IBPB before exit to userspace

==============================
Versions of relevant libraries
==============================
[pip3] flashinfer-python==0.6.6
[pip3] numpy==2.2.6
[pip3] nvidia-cublas-cu12==12.8.4.1
[pip3] nvidia-cuda-cupti-cu12==12.8.90
[pip3] nvidia-cuda-nvrtc-cu12==12.8.93
[pip3] nvidia-cuda-runtime-cu12==12.8.90
[pip3] nvidia-cudnn-cu12==9.10.2.21
[pip3] nvidia-cudnn-frontend==1.18.0
[pip3] nvidia-cufft-cu12==11.3.3.83
[pip3] nvidia-cufile-cu12==1.13.1.3
[pip3] nvidia-curand-cu12==10.3.9.90
[pip3] nvidia-cusolver-cu12==11.7.3.90
[pip3] nvidia-cusparse-cu12==12.5.8.93
[pip3] nvidia-cusparselt-cu12==0.7.1
[pip3] nvidia-cutlass-dsl==4.4.2
[pip3] nvidia-cutlass-dsl-libs-base==4.4.2
[pip3] nvidia-ml-py==13.595.45
[pip3] nvidia-nccl-cu12==2.27.5
[pip3] nvidia-nvjitlink-cu12==12.8.93
[pip3] nvidia-nvshmem-cu12==3.4.5
[pip3] nvidia-nvtx-cu12==12.8.90
[pip3] pyzmq==27.1.0
[pip3] torch==2.10.0+cu128
[pip3] torch-c-dlpack-ext==0.1.5
[pip3] torchaudio==2.10.0+cu128
[pip3] torchvision==0.25.0+cu128
[pip3] transformers==4.57.6
[pip3] triton==3.6.0
[conda] Could not collect

==============================
         vLLM Info
==============================
ROCM Version                 : Could not collect
vLLM Version                 : 0.18.1rc1.dev203+g97d19197b (git sha: 97d19197b)
vLLM Build Flags:
  CUDA Archs: Not Set; ROCm: Disabled
GPU Topology:
        GPU0    NIC0    NIC1    CPU Affinity    NUMA Affinity   GPU NUMA ID
GPU0     X      NODE    NODE    32-63,96-127    1               N/A
NIC0    NODE     X      PIX
NIC1    NODE    PIX      X

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

NIC Legend:

  NIC0: mlx5_0
  NIC1: mlx5_1
```
</details>

## Test Set Description

For publicly available AI Hub datasets, orthographic transcriptions were used as references. The test subsets follow the sample definitions from here
- CV 15: Korean test set of [Common Voice](https://commonvoice.mozilla.org/ko)
- FLEURS: Korean test set of [FLEURS](https://huggingface.co/datasets/google/fleurs)
- Callcenter: 3,000 samples from the validation split of [Low-Quality Telephone Speech Recognition Data]((https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=571)). The sample list is available [here](https://github.com/rtzr/Awesome-Korean-Speech-Recognition/blob/main/docs/AIHUB_TELEPHONE_LOW_QUALITY_test.txt)
- Callcenter2: 3,000 samples from the validation split of [Counseling Speech](https://www.aihub.or.kr/aihubdata/data/view.do?&dataSetSn=100). This dataset contains telephone-quality recordings from a variety of counseling scenarios. The sample list is available [here](https://github.com/rtzr/Awesome-Korean-Speech-Recognition/blob/main/docs/AIHUB_COUNSELING_test.txt)
- Conference: 3,000 samples from the validation split of [Conference Speech](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=132). The audio is excerpted from EBS broadcasts such as talk shows, debates, and news. The sample list is available [here](https://github.com/rtzr/Awesome-Korean-Speech-Recognition/blob/main/docs/AIHUB_CONFERENCE_CALL_test.txt) 
- Lecture: 3,000 samples from the validation split of [Korean Lecture](https://github.com/rtzr/Awesome-Korean-Speech-Recognition/blob/main/docs/AIHUB_KOREAN_LECTURE_test.txt). The audio is excerpted from EBS educational broadcasts. The sample list is available [here](https://github.com/rtzr/Awesome-Korean-Speech-Recognition/blob/main/docs/AIHUB_KOREAN_LECTURE_test.txt)
- Kspon-{clean,other}: Eval-Clean and Eval-Other subset from [Korean Speech]()