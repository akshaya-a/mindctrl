ARG BUILD_FROM
FROM $BUILD_FROM

COPY requirements.txt /
# TODO: have the nice people at mlflow create a mlflow gateway image
RUN pip install --no-cache-dir -r /requirements.txt

# Copy data for add-on
COPY run.sh /
RUN chmod a+x /run.sh

COPY route-config.yaml /
COPY config.py /

CMD [ "/run.sh" ]
