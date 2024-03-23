ARG BUILD_FROM="ghcr.io/hassio-addons/debian-base:7.3.3"
FROM $BUILD_FROM

# ensure local python is preferred over distribution python
ENV PATH /usr/local/bin:$PATH

# http://bugs.python.org/issue19846
# > At the moment, setting "LANG=C" on a Linux system *fundamentally breaks Python 3*, and that's not OK.
ENV LANG C.UTF-8
ENV MINDCTRL_ADDON_VERSION ${BUILD_VERSION}

# runtime dependencies
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    netbase \
    tzdata \
    python3-full \
    python3-pip \
    wget \
    ; \
    rm -rf /var/lib/apt/lists/*; \
    \
    python3 --version

RUN pip3 --version

COPY requirements.txt /

RUN pip3 install --no-cache --break-system-packages -r /requirements.txt


#   mlflow deployments start-server --config-path /usr/bin/deployment-server/route-config.yaml --port 5001 --host 0.0.0.0
COPY route-config.yaml /
COPY run_deployments.sh /
ENTRYPOINT [ "/bin/bash" ] # To disable s6 overlay
CMD [ "/run_deployments.sh"]


LABEL \
    maintainer="AK" \
    org.opencontainers.image.title="${BUILD_NAME}" \
    org.opencontainers.image.description="${BUILD_DESCRIPTION}" \
    org.opencontainers.image.vendor="AK" \
    org.opencontainers.image.authors="AK" \
    org.opencontainers.image.licenses="Apache 2.0" \
    org.opencontainers.image.url="https://addons.community" \
    org.opencontainers.image.source="https://github.com/${BUILD_REPOSITORY}" \
    org.opencontainers.image.documentation="https://github.com/${BUILD_REPOSITORY}/blob/main/README.md" \
    org.opencontainers.image.created=${BUILD_DATE} \
    org.opencontainers.image.revision=${BUILD_REF} \
    org.opencontainers.image.version=${BUILD_VERSION}
