"""Manage API keys from the command line.

    python -m agent.admin create-key --name demo --budget 1.0 --rpm 10
    python -m agent.admin create-key --name ops --admin
    python -m agent.admin list
    python -m agent.admin revoke key_ab12cd34ef

The data directory is AGENT_DATA_DIR (default ./data/agent). A new key is printed once and
cannot be shown again, because only its hash is stored.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from agent.accounts import Accounts


def accounts_db() -> Path:
    data = Path(os.environ.get("AGENT_DATA_DIR", "./data/agent"))
    data.mkdir(parents=True, exist_ok=True)
    return data / "accounts.db"


def main(argv: list[str] | None = None, out=None) -> int:
    out = out or sys.stdout
    parser = argparse.ArgumentParser(prog="agent.admin", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    create = commands.add_parser("create-key", help="make a new API key")
    create.add_argument("--name", required=True)
    create.add_argument("--budget", type=float, default=1.0, help="daily budget in US dollars")
    create.add_argument("--rpm", type=int, default=10, help="requests per minute")
    create.add_argument("--admin", action="store_true", help="may read /metrics")

    commands.add_parser("list", help="list keys and what they spent today")
    revoke = commands.add_parser("revoke", help="stop a key from working")
    revoke.add_argument("key_id")

    args = parser.parse_args(argv)
    accounts = Accounts(accounts_db())

    if args.command == "create-key":
        plain, key = accounts.create_key(args.name, args.budget, args.rpm, args.admin)
        print(f"id:   {key.id}\nkey:  {plain}\n(shown once, store it now)", file=out)
        return 0

    if args.command == "list":
        rows = accounts._db.execute(
            "SELECT id, name, daily_budget_usd, rpm, active, is_admin FROM api_keys ORDER BY created"
        ).fetchall()
        for key_id, name, budget, rpm, active, is_admin in rows:
            spent = accounts.spent_today(key_id)
            flags = ("admin " if is_admin else "") + ("" if active else "revoked")
            print(f"{key_id}  {name:20} ${spent:.4f} of ${budget:.2f} today  {rpm}/min  {flags}".rstrip(), file=out)
        return 0

    if not accounts.revoke(args.key_id):
        print(f"no key with id {args.key_id}", file=out)
        return 1
    print(f"revoked {args.key_id}", file=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
