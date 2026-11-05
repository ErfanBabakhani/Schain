"""Build the local server-connected Graph Explorer."""
import argparse
import hashlib
import shutil
import socket
from importlib.resources import files
from pathlib import Path

def build(output):
    output = Path(output).resolve()
    source = Path(str(files("visualization"))).resolve()
    if output == source or output in source.parents or source in output.parents:
        raise ValueError("Build output must be separate from visualization source")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source / "assets", output / "assets", dirs_exist_ok=True)
    html = (source / "index.html").read_text(encoding="utf-8")
    content = html + ''.join(p.read_text(encoding="utf-8") for p in sorted((source / "assets").rglob("*")) if p.is_file())
    revision = hashlib.sha256(content.encode()).hexdigest()[:12]
    html = html.replace("__GRAPH_DATA__", '{"nodes":[],"edges":[]}')
    for asset in (source / "assets").rglob("*"):
        if asset.is_file():
            relative = asset.relative_to(source).as_posix()
            html = html.replace(f'"{relative}"', f'"{relative}?v={revision}"')
    html = html.replace('<title>', f'<meta name="schain-frontend" content="projects-{revision}">\n  <title>', 1)
    (output / "index.html").write_text(html, encoding="utf-8")
    legacy = output / "explorer.html"
    if legacy.is_file() and 'Evidence explorer</title>' in legacy.read_text(encoding="utf-8")[:1000]:
        legacy.unlink()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = build(args.output)
    print(f"Built SChain project Graph Explorer: {output / 'index.html'}")
    with socket.socket() as probe:
        probe.settimeout(0.25)
        if probe.connect_ex(("127.0.0.1", 8080)) == 0:
            print("Port 8080 is already in use. Check that it serves this directory; stop the identified old static server or use port 8081 with matching SCHAIN_CORS_ORIGINS.")
    print(f'python -m http.server 8080 --bind 127.0.0.1 --directory "{output}"')


if __name__ == "__main__":
    main()
