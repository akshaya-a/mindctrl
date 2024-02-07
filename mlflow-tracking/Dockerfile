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

RUN python3 -m venv /venv
ENV PATH /venv/bin:$PATH

COPY requirements.txt /

RUN pip install --no-cache -r /requirements.txt

# Copy data for add-on
COPY run.sh /
RUN chmod a+x /run.sh

CMD [ "/run.sh" ]
