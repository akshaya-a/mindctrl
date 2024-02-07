ARG BUILD_FROM
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
# TODO: have the nice people at mlflow create a mlflow gateway image
RUN pip install --no-cache-dir -r /requirements.txt

# Copy data for add-on
COPY run.sh /
RUN chmod a+x /run.sh

COPY route-config.yaml /
# Patch psutil as a temporary fix for lack of aarch64 wheel
COPY utils.py.patch /venv/lib/python3.11/site-packages/mlflow/gateway/utils.py

CMD [ "/run.sh" ]
