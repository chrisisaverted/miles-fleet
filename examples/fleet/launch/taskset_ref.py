"""Parse a Fleet Registry taskset reference into its repository coordinate."""

import argparse
import re

_REGISTRY_PREFIX = "registry-alpha.fleetai.me/"
_SHA256_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


def parse_taskset_repository(taskset_ref: str) -> tuple[str, str]:
    """Return ``(namespace, repository)`` for a tagged or digest-pinned ref."""
    ref = taskset_ref.removeprefix("flt://").removeprefix(_REGISTRY_PREFIX)

    repository, digest_separator, digest = ref.partition("@")
    if digest_separator:
        if not _SHA256_DIGEST.fullmatch(digest):
            raise ValueError("taskset digest must be exactly sha256:<64 lowercase hex characters>")
    else:
        repository, tag_separator, tag = repository.rpartition(":")
        if not tag_separator:
            repository = ref
        elif not tag:
            raise ValueError("taskset tag must not be empty")

    parts = repository.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError("taskset reference must contain exactly namespace/repository")
    return parts[0], parts[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("taskset_ref")
    args = parser.parse_args()
    namespace, repository = parse_taskset_repository(args.taskset_ref)
    print(f"{namespace}/{repository}")


if __name__ == "__main__":
    main()
