"""
run.py — Bike Parking Buddy development server launcher.

Usage:
    python run.py               # default dev mode on port 8000
    python run.py --port 9000   # custom port
    python run.py --prod        # production settings (no reload)

Equivalent to:
    uvicorn backend.main:app --reload --port 8000
"""

import argparse
import sys
import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bike Parking Buddy — FastAPI dev server"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host (default: 127.0.0.1; use 0.0.0.0 to expose on network)",
    )
    parser.add_argument(
        "--prod",
        action="store_true",
        help="Run in production mode (no auto-reload, multiple workers)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of uvicorn workers (prod only; default: 1)",
    )
    args = parser.parse_args()

    print(f"""
+------------------------------------------------------+
|        Bike Parking Buddy API Server                 |
+------------------------------------------------------+
|  Host   : {args.host:<42}|
|  Port   : {str(args.port):<42}|
|  Mode   : {'production' if args.prod else 'development (auto-reload)':<42}|
|  Docs   : http://{args.host}:{args.port}/docs{'':<24}|
+------------------------------------------------------+
    """)

    config = dict(
        app="backend.main:app",
        host=args.host,
        port=args.port,
        reload=not args.prod,
        log_level="info",
    )
    if args.prod:
        config["workers"] = args.workers

    uvicorn.run(**config)


if __name__ == "__main__":
    main()
