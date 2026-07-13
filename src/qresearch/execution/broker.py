"""Broker abstraction. A single interface so strategies stay broker-agnostic;
concrete adapters (Binance/ccxt, Zerodha Kite, Alpaca) implement it. Consolidates
the scattered ``place_order``/execution snippets. Live adapters are stubs — wire
credentials via ``config.settings`` and never hard-code keys."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Order:
    symbol: str
    side: str          # "buy" | "sell"
    qty: float
    order_type: str = "market"
    price: float | None = None


class BrokerBase(ABC):
    @abstractmethod
    def place_order(self, order: Order) -> dict: ...
    @abstractmethod
    def get_positions(self) -> dict: ...


@dataclass
class PaperBroker(BrokerBase):
    """In-memory paper-trading broker for backtests / dry runs."""
    cash: float = 100_000.0
    positions: dict = field(default_factory=dict)
    fills: list = field(default_factory=list)

    def place_order(self, order: Order) -> dict:
        px = order.price or 0.0
        signed = order.qty if order.side == "buy" else -order.qty
        self.positions[order.symbol] = self.positions.get(order.symbol, 0) + signed
        self.cash -= signed * px
        fill = {"symbol": order.symbol, "side": order.side, "qty": order.qty, "price": px}
        self.fills.append(fill)
        return fill

    def get_positions(self) -> dict:
        return dict(self.positions)


class CcxtBroker(BrokerBase):
    """Live broker adapter over any ccxt-supported exchange (Binance, OKX,
    etc.) — consolidates the notebooks' hand-rolled per-exchange bot classes
    (``submit_market_order``/``confirm_order_filled``/``get_position``/
    ``get_account``/``cancel_all_orders``/``round_price``/``round_quantity``
    from ``data-cleaning_001``/``008``, which duplicated the same logic once
    for Alpaca and once for python-binance). ccxt is already a project
    dependency and covers both exchanges through one interface, so this
    replaces both — an Alpaca-specific adapter (a stock, not crypto, broker)
    is not added since ``alpaca-py`` isn't a project dependency; add one if
    that need materializes. Credentials come from ``config.settings``,
    never hard-coded."""

    def __init__(self, exchange_id: str, api_key: str = "", api_secret: str = ""):
        import ccxt  # local import: optional dependency
        self.exchange = getattr(ccxt, exchange_id)({"apiKey": api_key, "secret": api_secret})
        self.exchange.load_markets()

    def round_price(self, symbol: str, price: float) -> float:
        return float(self.exchange.price_to_precision(symbol, price))

    def round_quantity(self, symbol: str, qty: float) -> float:
        return float(self.exchange.amount_to_precision(symbol, qty))

    def place_order(self, order: Order) -> dict:
        qty = self.round_quantity(order.symbol, order.qty)
        if order.order_type == "market":
            return self.exchange.create_order(order.symbol, "market", order.side, qty)
        price = self.round_price(order.symbol, order.price or 0.0)
        return self.exchange.create_order(order.symbol, order.order_type, order.side, qty, price)

    def confirm_order_filled(self, symbol: str, order_id: str, timeout: float = 5.0, poll: float = 0.5) -> bool:
        import time
        start = time.time()
        while time.time() - start < timeout:
            status = str(self.exchange.fetch_order(order_id, symbol).get("status", "")).lower()
            if status == "closed":
                return True
            if status in ("canceled", "cancelled", "rejected", "expired"):
                return False
            time.sleep(poll)
        return False

    def get_positions(self) -> dict:
        try:
            return {p["symbol"]: p for p in self.exchange.fetch_positions() if p.get("contracts")}
        except Exception:
            balances = self.exchange.fetch_balance().get("total", {})
            return {k: v for k, v in balances.items() if v}

    def get_account(self) -> dict | None:
        try:
            return self.exchange.fetch_balance()
        except Exception:
            return None

    def cancel_all_orders(self, symbol: str) -> bool:
        try:
            for o in self.exchange.fetch_open_orders(symbol):
                self.exchange.cancel_order(o["id"], symbol)
            return True
        except Exception:
            return False
