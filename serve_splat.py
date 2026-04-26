import argparse
import http.server
import os
import socket
import webbrowser
from functools import partial


def parse_args():
    parser = argparse.ArgumentParser(description="Serve the splat web viewer locally")
    parser.add_argument(
        "--dir",
        default="viewer",
        help="Directory containing the splat viewer files (default: viewer)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to listen on (default: 7860)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open the browser automatically",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    serve_dir = os.path.abspath(args.dir)
    if not os.path.isdir(serve_dir):
        raise FileNotFoundError(
            f"Viewer directory not found: {serve_dir}\n"
            "Clone it first:  git clone https://github.com/camenduru/splat viewer"
        )

    os.chdir(serve_dir)

    handler = partial(http.server.SimpleHTTPRequestHandler, directory=serve_dir)

    # Find a free port if the requested one is busy
    port = args.port
    for attempt in range(10):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
            break
        except OSError:
            port += 1
    else:
        raise RuntimeError("Could not find a free port in range "
                           f"{args.port}–{args.port+9}")

    url = f"http://localhost:{port}"
    print(f"Serving splat viewer at  {url}")
    print(f"Serving files from       {serve_dir}")
    print("Press Ctrl+C to stop.\n")

    if not args.no_browser:
        webbrowser.open(url)

    with http.server.HTTPServer(("", port), handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
