# miles trainer image for Fleet training on rl1.
#
# Base is the official miles image (pinned sglang/Megatron/kernels). The cu12
# variant matches the cluster's driver generation. On top: this repo's tree
# (replacing the image's /root/miles checkout), the fleet-runtime SDK and the
# flt binary from the SAME platform commit (client and SDK must agree on the
# taskset schema), and a docker CLI (the daemon is the pod's dind sidecar).
#
# Build from the REPO ROOT with a token that reads fleet-ai/platform:
#   DOCKER_BUILDKIT=1 docker build -f examples/fleet/launch/rl1/Dockerfile \
#     --secret id=gh_token,env=GH_TOKEN .

FROM golang:1.26 AS flt-builder
ARG PLATFORM_REF=72e656c948ab3dbc018c844b76b93bfb297f2061
RUN --mount=type=secret,id=gh_token \
    git config --global url."https://x-access-token:$(cat /run/secrets/gh_token)@github.com/".insteadOf "https://github.com/" \
    && git clone --filter=blob:none https://github.com/fleet-ai/platform /src \
    && cd /src && git checkout -q ${PLATFORM_REF} \
    && cd registry && CGO_ENABLED=0 go build -o /flt ./cmd/flt

FROM radixark/miles:dev-cu12
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
    && python -c "import examples.fleet.rollout, examples.fleet.session, examples.fleet.prepare_dataset; print('fleet integration imports OK')" \
    && python -c "import fleet_runtime; from fleet_runtime.cli.sources import resolve_source; print('fleet_runtime OK')" \
    && python -c "from miles.utils.chat_template_utils.tito_tokenizer import GLM47TITOTokenizer; print('tito OK')" \
    && (flt version || flt --version) \
    && docker --version \
    && hf version || hf --help >/dev/null
