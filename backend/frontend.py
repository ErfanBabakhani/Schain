"""Build the local server-connected Graph Explorer."""
import argparse
import hashlib
import socket
from importlib.resources import files
from pathlib import Path

ASSET_FILES = (
    "assets/css/styles.css", "assets/css/projects.css",
    "assets/js/explorer.js", "assets/js/projects.js",
)


def render_frontend():
    source = files("visualization")
    html = source.joinpath("index.html").read_text(encoding="utf-8")
    if 'data-mode="server"' not in html or 'id="projectSelect"' not in html or 'demo-data.js' in html:
        raise ValueError("Frontend template must be the server project workspace")
    assets = {name: source.joinpath(name).read_bytes() for name in ASSET_FILES}
    digest = hashlib.sha256(html.encode("utf-8"))
    for name, content in assets.items():
        digest.update(name.encode("ascii") + b"\0" + content)
    revision = digest.hexdigest()[:12]
    html = html.replace("__GRAPH_DATA__", '{"nodes":[],"edges":[]}')
    for name in assets:
        html = html.replace(f'"{name}"', f'"{name}?v={revision}"')
    html = html.replace('<title>', f'<meta name="schain-frontend" content="projects-{revision}">\n  <title>', 1)
    return revision, {"index.html": html.encode("utf-8"), **assets}


def build(output):
    output = Path(output).resolve()
    source = Path(str(files("visualization"))).resolve()
    if output == source or output in source.parents or source in output.parents:
        raise ValueError("Build output must be separate from visualization source")
    output.mkdir(parents=True, exist_ok=True)
    _, assets = render_frontend()
    for name, content in assets.items():
        target = output / name
        if not target.resolve().is_relative_to(output):
            raise ValueError("Build assets must stay inside the output directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file() or target.read_bytes() != content:
            target.write_bytes(content)
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
