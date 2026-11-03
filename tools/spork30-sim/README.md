# spork30 5-minute maker: simulation dashboard (Railway)

Runs `python -m spork30.dashboard` from github.com/0xbrutusdev/polymarket-5min-crypto-trading-bot
(pinned audited commit, see Dockerfile) and shows its terminal screen read-only in the browser (ttyd + tmux).

- **Simulation only.** start.sh unsets POLYMARKET_PRIVATE_KEY / LIVE_TRADING_CONFIRM / POLYMARKET_FUNDER and forces
  TRADING_MODE=SIMULATION, so no order can be signed even if those variables are added to the service.
- It quotes **international Polymarket** BTC 5-minute markets (polymarket.com CLOB), which US users cannot trade.
  It is a watch-only reference next to the Polymarket US bot, like the 5 MIN card.
- Runs in a non-US Railway region because api.binance.com / stream.binance.com return 451 to US IPs.
