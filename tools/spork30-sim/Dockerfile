# spork30 maker bot dashboard (github.com/0xbrutusdev/polymarket-5min-crypto-trading-bot),
# SIMULATION ONLY, shown read-only in the browser through ttyd + tmux.
# The upstream code is cloned at an audited commit at build time (it has no license, so it is not vendored here).
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git tmux curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*
ARG TTYD_VERSION=1.7.7
RUN curl -fsSL -o /usr/local/bin/ttyd https://github.com/tsl0922/ttyd/releases/download/${TTYD_VERSION}/ttyd.x86_64 \
 && chmod +x /usr/local/bin/ttyd
# Audited 2026-11-03: only polymarket.com / binance.com hosts, the key goes only to the official py-clob-client-v2.
ARG SPORK_COMMIT=b63e2990f8352e51efc9622fe73595f66f750f3d
RUN git clone https://github.com/0xbrutusdev/polymarket-5min-crypto-trading-bot /app \
 && cd /app && git checkout -q ${SPORK_COMMIT} && rm -rf .git \
 && find . -name __pycache__ -prune -exec rm -rf {} +
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt
COPY start.sh /start.sh
RUN chmod +x /start.sh
ENV TERM=xterm-256color PYTHONUNBUFFERED=1
CMD ["/start.sh"]
