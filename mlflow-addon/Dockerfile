ARG BUILD_FROM="ghcr.io/hassio-addons/debian-base:7.2.0"
FROM $BUILD_FROM

# ensure local python is preferred over distribution python
ENV PATH /usr/local/bin:$PATH

# http://bugs.python.org/issue19846
# > At the moment, setting "LANG=C" on a Linux system *fundamentally breaks Python 3*, and that's not OK.
ENV LANG C.UTF-8

# runtime dependencies
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    netbase \
    tzdata \
    python3-full \
    python3-pip \
    ; \
    rm -rf /var/lib/apt/lists/*; \
    \
    python3 --version

RUN pip3 --version

COPY requirements.txt /

RUN pip3 install --no-cache --break-system-packages -r /requirements.txt

COPY rootfs /

RUN chmod a+x /usr/bin/run_tracking.sh
RUN chmod a+x /usr/bin/run_gateway.sh
RUN chmod a+x /usr/bin/run_multiserver.sh
RUN chmod a+x /etc/s6-overlay/s6-rc.d/init-server-env/run
RUN chmod a+x /etc/s6-overlay/s6-rc.d/gateway/data/check

# Patch psutil as a temporary fix for lack of aarch64 wheel
COPY utils.py.patch /usr/local/lib/python3.11/dist-packages/mlflow/gateway/utils.py

LABEL \
    io.hass.name="${BUILD_NAME}" \
    io.hass.description="${BUILD_DESCRIPTION}" \
    io.hass.arch="${BUILD_ARCH}" \
    io.hass.type="addon" \
    io.hass.version=${BUILD_VERSION} \
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
