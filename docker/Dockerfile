# Базовый образ
FROM python:3.10.8-buster as python-base

# Настройки окружения
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=2.0.0 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PYSETUP_PATH="/opt/pysetup" \
    VENV_PATH="/opt/pysetup/.venv"

# Добавляем Poetry и виртуальное окружение в PATH
ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

# Устанавливаем системные зависимости и Poetry
FROM python-base as builder-base
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Установка Poetry
RUN curl -sSL https://install.python-poetry.org | python -

# Копируем файлы зависимостей
WORKDIR $PYSETUP_PATH
COPY poetry.lock pyproject.toml README.md ./

# Устанавливаем зависимости
RUN poetry install --no-root



