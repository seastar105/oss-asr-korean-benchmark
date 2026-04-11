# Korean OSS ASR Benchmark

2026년 3월 기준으로, 한국어를 공식적으로 지원하는 여러 음성인식 모델들이 출시되었습니다만, 사용하는 입장에서 제일 궁금한 질문은 `whisper보다 좋은가?`입니다.

다만, 공개하는 회사에서는 Common Voice, FLEURS 정도의 테스트셋에 대해서만 결과를 쓰거나, 아예 한국어 성능은 빠져있다보니 알기 어려워서 AI Hub에 공개된 데이터셋을 사용해서 테스트를 수행한 결과입니다.

## 업데이트

2026.04.11: FunAudioLLM/SenseVoiceSmall, RaonSpeech/Raon-Speech-9B를 테이블에 추가했습니다. CER 계산시에 텍스트 정규화 방식을 바꿨습니다.

## 테스트 결과 (CER, RTF)

NOTE: 테스트 음성들이 모두 30초 이하의 짧은 음성이기 때문에 이 결과가 long-form 음성에 대한 결과로 이어지진 않습니다.

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

Raon-Speech-9B 모델은 다른 모델과 달리 RTX 5090 모델에서 돌리는 것에 실패하여 RTX Pro 6000 Blackwell을 사용하여 평가를 수행했습니다.

SenseVoiceSmall 모델의 RTF는 배치 추론을 사용한 경우입니다.

**NOTE: FLEURS, Ksponspeech 테스트셋에 대해서는 Raon-Speech-9B의 테크니컬 리포트에서 보고된 CER 값과 다릅니다. 아마 reference text가 달라서 그런 것으로 추정합니다.**

whisper-large-v3이 여전히 평균 성능이 제일 좋으며, 특히 `Callcenter`에서 큰 차이가 납니다.

whisper-medium-komixv2의 경우 한국어 파인튜닝을 수행한 모델로, 음성인식 모델을 특정 도메인에 적용하고자 할 경우 여전히 파인튜닝이 굉장히 유효하다는걸 보여줍니다. 

특히, 음성 도메인이 다른 도메인과 다른 `Callcenter`, 텍스트 도메인이 다른 도메인과 다른 `KsponSpeech`에서 효과가 크기 때문에 운용 코스트는 RTF가 낮은 모델에 도메인 데이터로 파인튜닝을 하는 것이 낮을 것 같습니다.

## 실행환경

모든 평가는 vllm을 사용하여 서버를 실행(`vllm serve <model-name>`)한 뒤 32개 스레드로 리퀘스트를 보내 음성인식 결과를 얻는 형식으로 수행되었으며, [run_asr.py](./run_asr.py)와 [run_asr_stream.py](./run_asr_stream.py)를 참조해주세요.

실행 서버스펙은 아래와 같습니다.

<details>

<summary>실행환경</summary>

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

## 테스트셋 소개

AI Hub 공개 데이터들에 대해서는 철자 전사를 reference로 사용하였으며, 테스트셋은 [이곳](https://github.com/rtzr/Awesome-Korean-Speech-Recognition)에 정의된 샘플들을 사용했습니다.

- CV 15: [Common Voice](https://commonvoice.mozilla.org/ko)의 한국어 테스트 셋
- FLEURS: [FLEURS](https://huggingface.co/datasets/google/fleurs)의 한국어 테스트셋
- Callcenter: [저음질 전화망 음성인식 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=571)의 Validation set 중 3000개 샘플 사용. 목록은 [이곳](https://github.com/rtzr/Awesome-Korean-Speech-Recognition/blob/main/docs/AIHUB_TELEPHONE_LOW_QUALITY_test.txt) 참조.
- Callcenter2: [상담 음성](https://www.aihub.or.kr/aihubdata/data/view.do?&dataSetSn=100)의 Validation set 중 3000개 샘플 사용. 다양한 상담 음성 시나리오를 전화 품질로 녹음한 음성입니다. 목록은 [이곳](https://github.com/rtzr/Awesome-Korean-Speech-Recognition/blob/main/docs/AIHUB_COUNSELING_test.txt) 참조. 
- Conference: [회의 음성](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=132)의 Validation set 중 3000개 샘플 사용. EBS 방송 중 토크, 토론, 뉴스 등의 음성에서 발췌한 음성입니다. 목록은 [이곳](https://github.com/rtzr/Awesome-Korean-Speech-Recognition/blob/main/docs/AIHUB_CONFERENCE_CALL_test.txt) 참조
- Lecture: [한국어 강의](https://github.com/rtzr/Awesome-Korean-Speech-Recognition/blob/main/docs/AIHUB_KOREAN_LECTURE_test.txt)의 Validation set 중 3000개 샘플 사용. EBS 교육방송에서 발췌한 음성입니다. 목록은 [이곳](https://github.com/rtzr/Awesome-Korean-Speech-Recognition/blob/main/docs/AIHUB_KOREAN_LECTURE_test.txt) 참조
- Kspon-{clean,other}: [한국어 음성](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=123) 테스트 셋인 Eval-Clean과 Eval-Other입니다.