# Source-build provider selection follow-up

The first pass used a clean SGLang source checkout at commit 5792cf6e5e11aebc62d41cf19828f615a8fa7e0a. Its placeholder __version__ of 0.0.0 caused the release-floor gate to skip compatible source kernels. The corrected probe records the actual imported checkout as git:<commit> (tracked changes receive a diff hash). It does not invent a release number; existing provider logic treats a non-release identifier as unclassified. Every kernel still passes its shape/state gates and numerical cross-check.

The actual SGLang fused KDA kernel passed cross-check at B1 and B8: p50 10.848/13.632 us (exact values are authoritative in results/sglang-source-kda.csv). The first pass's Torch eager KDA implementation was 245.568/266.016 us. Torch eager B1 KDA alone contributed 69 * 245.568 us = 16.944192 ms to the overlay. This establishes a provider-selection problem, not a GPU slowdown. Fourteen additional SGLang quantisation/activation measurements passed cross-check in the targeted follow-up; previously measured SGLang implementations were supplied as measured-in inputs to avoid repeats.

The SGL kernel routing provider remains gated because installed sgl_kernel 0.4.10 is outside that adapter's <0.4.5 API contract. This bound is retained; it requires a compatible adapter, not relabeling the version.

Correctness of a slow eager reference is not sufficient to make its timing representative of production SGLang. The data and report retain this limitation. These L2-drain isolated timings also do not capture warm graph / PDL overlap.
