# miles trainer image for Fleet training.
#
# Base is the official miles image (pinned sglang/Megatron/kernels). The cu12
# variant matches the cluster's driver generation. On top: this repo's tree
# (replacing the image's /root/miles checkout), the fleet-runtime SDK and the
# flt binary from the SAME platform commit (client and SDK must agree on the
# taskset schema), and a docker CLI (the daemon is the pod's dind sidecar).
#
# Build from the REPO ROOT with a token that reads fleet-ai/platform:
#   DOCKER_BUILDKIT=1 docker build -f examples/fleet/launch/Dockerfile \
#     --secret id=gh_token,env=GH_TOKEN .

FROM golang:1.26 AS flt-builder
ARG PLATFORM_REF=72e656c948ab3dbc018c844b76b93bfb297f2061
RUN --mount=type=secret,id=gh_token \
    git config --global url."https://x-access-token:$(cat /run/secrets/gh_token)@github.com/".insteadOf "https://github.com/" \
    && git clone --filter=blob:none https://github.com/fleet-ai/platform /src \
    && cd /src && git checkout -q ${PLATFORM_REF} \
    && cd registry && CGO_ENABLED=0 go build -o /flt ./cmd/flt

# glm53-bu branch: base is the GLM-5.3 pairing image (miles + the
# sglang-miles-glm53next branch carrying the GLM-5.3 model port and its
# image processor, sgl-project/sglang#36833), digest-pinned because the tag
# moves. This tree replaces /root/miles with our branch, which includes
# upstream PR #2792 (GLM-5.3 vision RL). Move the pin only together with a
# debug_minimal smoke.
FROM radixark/miles@sha256:66725f740a6013b00d27e21fdfd480a24b0d3b5c61840342e9405bf1e09e5162
ARG PLATFORM_REF=72e656c948ab3dbc018c844b76b93bfb297f2061

# docker CLI only (daemon = dind sidecar via DOCKER_HOST)
RUN curl -fsSL https://download.docker.com/linux/static/stable/x86_64/docker-27.5.1.tgz \
    | tar -xz --strip-components=1 -C /usr/local/bin docker/docker

# This repo replaces the image's miles checkout; same package layout, so the
# editable install keeps working, refreshed below.
RUN rm -rf /root/miles
COPY . /root/miles
RUN cd /root/miles && pip install -e . --no-deps

# fleet-runtime pinned to the exact commit the flt binary is built from.
RUN --mount=type=secret,id=gh_token \
    git config --global url."https://x-access-token:$(cat /run/secrets/gh_token)@github.com/".insteadOf "https://github.com/" \
    && pip install "fleet-runtime @ git+https://github.com/fleet-ai/platform.git@${PLATFORM_REF}#subdirectory=runtime" \
    && (git config --global --unset-all url."https://x-access-token:$(cat /run/secrets/gh_token)@github.com/".insteadOf || true)

COPY --from=flt-builder /flt /usr/local/bin/flt
ENV FLEET_FLT=/usr/local/bin/flt

# GLM-5.3 vision processor overlay: the miles glm5_next plugin imports
# sglang.srt.configs.glm5_next_processing, which is added by sglang PR
# #36833 and is NOT in the glm53next base image (built 2026-08-27, before
# the PR). Overlay the PR's three package files, pinned to its head commit,
# until upstream rebuilds the image. Observed failure without this:
# ModuleNotFoundError at train start (miles-glm53-smoke, 2026-08-29).
# The KDA/Triton hotfix upstream PR #2792 names as an environment concern.
# The base image vendors an fla build whose KDA kernels call
# triton.next_power_of_2 inside the kernel, which this Triton refuses
# ("Unsupported function referenced", all 24 train ranks, 2026-08-29).
# The fix is fla-org/flash-linear-attention#1108 ("Hoist in-kernel
# triton.next_power_of_2 to host launchers"), on fla main and in NO PyPI
# release (0.5.2 does not even ship fla/ops/kda). Remove the vendored tree
# fully first: a pip swap leaves its untracked kda files behind.
RUN pip uninstall -y flash-linear-attention     && rm -rf /usr/local/lib/python3.12/dist-packages/fla     && pip install --no-deps --no-build-isolation        "git+https://github.com/fla-org/flash-linear-attention@ad4af37f03a90c38a15df50c6397a55594f5465e"     && python -c "import fla.ops.kda; print('fla kda ops OK, version', __import__('fla').__version__)"

ARG SGLANG_PR_SHA=21572b4470442fe49ea4ff36e64b1118b465c3b0
RUN SGL=$(python -c "import sglang,os;print(os.path.dirname(sglang.__file__))") \
    && for f in srt/configs/glm5_next.py srt/configs/glm5_next_processing.py srt/multimodal/processors/glm4v.py; do \
         curl -fsSL -o "$SGL/$f" "https://raw.githubusercontent.com/sgl-project/sglang/${SGLANG_PR_SHA}/python/sglang/$f" || exit 1; \
       done \
    && python -c "from sglang.srt.configs.glm5_next_processing import Glm5NextProcessor; print('glm5_next processor OK')"

# Off-by-one fix, pending upstream (sgl-project/sglang#36833): glm4v.py pairs
# the client's VERBATIM pretokenized input_ids (AVOID_RETOKENIZE) with an
# attention_mask computed over decode->retokenize text; any sampled token
# pair that is not canonical BPE (e.g. two newline tokens) makes the mask
# one short and get_rope_index_glm4v raises IndexError (observed at
# 8323/36643 tokens, miles-glm53-96k-probe 2026-08-29; local repro in
# glm-repro/repro.py). For a single unpadded sequence the correct mask is
# all-ones over the real ids, which the callee builds itself when None.
RUN python3 - <<'PATCH'
import sglang, os
p = os.path.join(os.path.dirname(sglang.__file__), "srt/multimodal/processors/glm4v.py")
s = open(p).read()
old = 'attention_mask=getattr(ret, "attention_mask", None),'
assert s.count(old) >= 1, "glm4v.py patch anchor not found"
s = s.replace(old, 'attention_mask=None,  # miles-fleet: ids are verbatim pretokenized; ret mask is over retokenized text (off-by-one)', 1)
open(p, "w").write(s)
print("glm4v.py off-by-one patched")
PATCH

# Import smoke: `pip install completed` != `image works`. Pulls the full
# integration import chain (miles + sglang + fleet_runtime + our modules).
RUN cd /root/miles \
    && python -c "import examples.fleet.rollout, examples.fleet.session, examples.fleet.prepare_dataset; import importlib.util as u, sys; spec = u.spec_from_file_location('rf', 'examples/fleet/launch/run_fleet.py'); m = u.module_from_spec(spec); spec.loader.exec_module(m); print('fleet integration + launcher import OK')" \
    && python -c "import fleet_runtime; from fleet_runtime.cli.sources import resolve_source; print('fleet_runtime OK')" \
    && python -c "from miles.utils.chat_template_utils.tito_tokenizer import GLM47TITOTokenizer; print('tito OK')" \
    && (flt version || flt --version) \
    && docker --version \
    && hf version || hf --help >/dev/null
