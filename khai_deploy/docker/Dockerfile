FROM python:3.12-slim
ARG KHAI_DEPLOY_VERSION
ARG KHAI_CORE_VERSION
LABEL org.opencontainers.image.version="${KHAI_DEPLOY_VERSION}" khai.core.version="${KHAI_CORE_VERSION}"
COPY dist/ /tmp/dist/
RUN pip install --no-cache-dir /tmp/dist/*.whl && rm -rf /tmp/dist
