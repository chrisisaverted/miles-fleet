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

# Pinned to a dated tag, never the moving dev-cu12 (upstream repushes it and
# a repush once shipped an sglang the then-current miles code could not
# start engines on). The pin and the miles merge-base move TOGETHER: this
# tag matches upstream main as of the 2026-08-28 fast-forward (sglang
# 0.5.18). Move the pin only together with a debug_minimal smoke.
FROM radixark/miles:dev-cu12-202608281709
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

# Import smoke: `pip install completed` != `image works`. Pulls the full
# integration import chain (miles + sglang + fleet_runtime + our modules).
RUN cd /root/miles \
    && python -c "import examples.fleet.rollout, examples.fleet.session, examples.fleet.prepare_dataset; import importlib.util as u, sys; spec = u.spec_from_file_location('rf', 'examples/fleet/launch/run_fleet.py'); m = u.module_from_spec(spec); spec.loader.exec_module(m); print('fleet integration + launcher import OK')" \
    && python -c "import fleet_runtime; from fleet_runtime.cli.sources import resolve_source; print('fleet_runtime OK')" \
    && python -c "from miles.utils.chat_template_utils.tito_tokenizer import GLM47TITOTokenizer; print('tito OK')" \
    && (flt version || flt --version) \
    && docker --version \
    && hf version || hf --help >/dev/null
