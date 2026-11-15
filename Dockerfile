FROM python:3.13-slim-bookworm AS build
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip wheel --no-cache-dir --wheel-dir /wheels .

FROM python:3.13-slim-bookworm AS runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /app
COPY --from=build /wheels /wheels
RUN python -m pip install --no-cache-dir --no-index --find-links=/wheels neuroevolution-simulator \
    && rm -rf /wheels \
    && groupadd --gid 10001 simulator \
    && useradd --uid 10001 --gid simulator --no-create-home simulator
COPY alembic.ini ./
COPY migrations ./migrations
USER simulator
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "neuroevolution.server.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
