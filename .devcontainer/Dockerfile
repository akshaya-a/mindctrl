FROM ghcr.io/home-assistant/devcontainer:addons

RUN apt-get update && apt-get install -y python3 python3-venv python3-pip

# https://code.visualstudio.com/remote/advancedcontainers/start-processes#_adding-startup-commands-to-the-docker-image-instead
COPY docker-entrypoint.sh /
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT [ "/docker-entrypoint.sh" ]
CMD [ "sleep", "infinity" ]
