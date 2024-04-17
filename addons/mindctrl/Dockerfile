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

RUN pip3 install --no-cache --break-system-packages uv

ENV SENTENCE_TRANSFORMERS_HOME=/model_cache
ENV HF_HUB_CACHE=/model_cache
# HF is super flaky today
ENV HF_HUB_DOWNLOAD_TIMEOUT=120
#COPY cache_models.py /cache_models.py
#RUN python3 /cache_models.py
#RUN huggingface-cli download sentence-transformers/all-MiniLM-L6-v2

ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1
ENV DO_NOT_TRACK=1

RUN mkdir /dapr; \
    wget -q https://raw.githubusercontent.com/dapr/cli/master/install/install.sh -O - | /bin/bash
RUN dapr init --slim

COPY rootfs /
RUN chmod a+x /.context/install_traefik.sh
ENV TRAEFIK_ARCH=${BUILD_ARCH}
RUN /.context/install_traefik.sh

COPY requirements.txt /
RUN uv pip install --no-cache --break-system-packages --python python3 -r /requirements.txt
# One day when builder workflow supports --build-context
# COPY --from=mindctrl_source / /pysrc
RUN uv pip install --no-cache --break-system-packages --python python3 /.context/pysrc/

RUN chmod a+x /usr/bin/run_tracking.sh && \
    chmod a+x /usr/bin/run_gateway.sh && \
    chmod a+x /usr/bin/run_multiserver.sh && \
    chmod a+x /usr/bin/run_traefik.sh && \
    chmod a+x /usr/bin/run_dashboard.sh && \
    chmod a+x /etc/s6-overlay/s6-rc.d/init-server-env/run && \
    chmod a+x /etc/s6-overlay/s6-rc.d/discovery/run && \
    chmod a+x /etc/s6-overlay/s6-rc.d/gateway/data/check && \
    chmod a+x /etc/s6-overlay/s6-rc.d/tracking/data/check


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
