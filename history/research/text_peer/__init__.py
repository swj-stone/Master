"""Explicit primary rules from the two frozen binary-text experiments."""
from research.text_peer.sparse_game import select as _select_log
from research.text_peer.square import choose as _select_square


def log_peer_support(X, y, *, count=None):
    return _select_log(X, y, count=count, rule="both", utility="log", scale="sqrt")


def square_peer_support(X, y, *, count=None):
    return _select_square(X, y, count=count, rule="both", absence=True)
