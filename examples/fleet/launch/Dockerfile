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
ARG SGLANG_PR_SHA=21572b4470442fe49ea4ff36e64b1118b465c3b0
RUN SGL=$(python -c "import sglang,os;print(os.path.dirname(sglang.__file__))") \
    && for f in srt/configs/glm5_next.py srt/configs/glm5_next_processing.py srt/multimodal/processors/glm4v.py; do \
         curl -fsSL -o "$SGL/$f" "https://raw.githubusercontent.com/sgl-project/sglang/${SGLANG_PR_SHA}/python/sglang/$f" || exit 1; \
       done \
    && python -c "from sglang.srt.configs.glm5_next_processing import Glm5NextProcessor; print('glm5_next processor OK')"

# Import smoke: `pip install completed` != `image works`. Pulls the full
# integration import chain (miles + sglang + fleet_runtime + our modules).
RUN cd /root/miles \
    && python -c "import examples.fleet.rollout, examples.fleet.session, examples.fleet.prepare_dataset; import importlib.util as u, sys; spec = u.spec_from_file_location('rf', 'examples/fleet/launch/run_fleet.py'); m = u.module_from_spec(spec); spec.loader.exec_module(m); print('fleet integration + launcher import OK')" \
    && python -c "import fleet_runtime; from fleet_runtime.cli.sources import resolve_source; print('fleet_runtime OK')" \
    && python -c "from miles.utils.chat_template_utils.tito_tokenizer import GLM47TITOTokenizer; print('tito OK')" \
    && (flt version || flt --version) \
    && docker --version \
    && hf version || hf --help >/dev/null
