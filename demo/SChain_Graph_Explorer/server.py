#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import html
import hashlib
import ipaddress
import json
import mimetypes
import re
import socket
import threading
from html.parser import HTMLParser
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
import xml.etree.ElementTree as ET
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
mimetypes.add_type("application/vnd.sqlite3", ".sqlite")
MAX_PREVIEW_BYTES = 5 * 1024 * 1024
SOURCE_CACHE_PATH = ROOT / "data" / "source_url_cache.json"
SOURCE_CACHE_LOCK = threading.Lock()
EVIDENCE_CONTENT_CACHE_PATH = ROOT / "data" / "evidence_content_cache.json"
EVIDENCE_CONTENT_CACHE_LOCK = threading.Lock()


def _load_source_cache() -> dict[str, str]:
    try:
        raw = json.loads(SOURCE_CACHE_PATH.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return {str(k): str(v) for k, v in raw.items() if k and v}
    except Exception:
        pass
    return {}


SOURCE_CACHE = _load_source_cache()


def _load_evidence_content_cache() -> dict[str, dict]:
    try:
        raw = json.loads(EVIDENCE_CONTENT_CACHE_PATH.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return {str(k): v for k, v in raw.items() if isinstance(v, dict)}
    except Exception:
        pass
    return {}


EVIDENCE_CONTENT_CACHE = _load_evidence_content_cache()


def _public_http_url(raw: str) -> str:
    parsed = urllib.parse.urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Only public HTTP/HTTPS URLs are supported.")
    host = parsed.hostname.lower()
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
        raise ValueError("Local addresses are not allowed.")
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
        for info in infos:
            addr = ipaddress.ip_address(info[4][0])
            if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_multicast or addr.is_reserved:
                raise ValueError("Private network addresses are not allowed.")
    except socket.gaierror as exc:
        raise ValueError("Could not resolve source host.") from exc
    return raw


def _is_google_news_url(raw: str) -> bool:
    try:
        parsed = urllib.parse.urlsplit(raw)
        return parsed.hostname == "news.google.com" and "/articles/" in parsed.path
    except Exception:
        return False


def _google_news_article_id(raw: str) -> str:
    parsed = urllib.parse.urlsplit(raw)
    parts = [p for p in parsed.path.split("/") if p]
    try:
        idx = parts.index("articles")
        return parts[idx + 1]
    except (ValueError, IndexError):
        return parts[-1] if parts else ""


def _legacy_google_news_decode(article_id: str) -> str | None:

    try:
        padded = article_id + "=" * (-len(article_id) % 4)
        raw = base64.urlsafe_b64decode(padded.encode("ascii"))
    except Exception:
        return None
    match = re.search(rb"https?://[^\x00-\x20\x7f]+", raw)
    if not match:
        return None
    candidate = match.group(0).decode("utf-8", errors="ignore")
    return candidate if candidate.startswith(("http://", "https://")) else None


def _attrs_from_tag(tag: str) -> dict[str, str]:
    attrs = {}
    pattern = re.compile(r"([\w:-]+)\s*=\s*([\"'])(.*?)\2", re.S)
    for key, _quote, value in pattern.findall(tag):
        attrs[key.lower()] = html.unescape(value)
    return attrs


def _fetch_google_params(article_id: str) -> tuple[str, str, str] | None:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/123 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.8",
    }
    for prefix in ("https://news.google.com/articles/", "https://news.google.com/rss/articles/"):
        try:
            req = urllib.request.Request(prefix + article_id, headers=headers)
            with urllib.request.urlopen(req, timeout=12) as resp:
                document = resp.read(2 * 1024 * 1024).decode(
                    resp.headers.get_content_charset() or "utf-8", errors="replace"
                )
            tags = re.findall(r"<[^>]*data-n-a-id=[^>]*>", document, flags=re.I)
            parsed_tags = [_attrs_from_tag(tag) for tag in tags]
            for attrs in parsed_tags:
                aid = attrs.get("data-n-a-id")
                sig = attrs.get("data-n-a-sg")
                ts = attrs.get("data-n-a-ts")
                if aid == article_id and sig and ts:
                    return aid, sig, ts
            for attrs in parsed_tags:
                aid = attrs.get("data-n-a-id")
                sig = attrs.get("data-n-a-sg")
                ts = attrs.get("data-n-a-ts")
                if aid and sig and ts:
                    return aid, sig, ts
        except Exception:
            continue
    return None


def _find_external_url(value, expected_domain: str = "") -> str | None:
    expected = expected_domain.lower().replace("www.", "").strip()
    found: list[str] = []

    def walk(obj, depth=0):
        if depth > 8:
            return
        if isinstance(obj, str):
            text = obj.strip()
            if text.startswith(("http://", "https://")):
                try:
                    host = (urllib.parse.urlsplit(text).hostname or "").lower().replace("www.", "")
                    if host and host not in {"news.google.com", "google.com"} and not host.endswith(".google.com"):
                        found.append(text)
                except Exception:
                    pass
            if text[:1] in "[{":
                try:
                    walk(json.loads(text), depth + 1)
                except Exception:
                    pass
        elif isinstance(obj, dict):
            for item in obj.values():
                walk(item, depth + 1)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                walk(item, depth + 1)

    walk(value)
    if not found:
        return None
    if expected:
        for candidate in found:
            host = (urllib.parse.urlsplit(candidate).hostname or "").lower().replace("www.", "")
            if host == expected or host.endswith("." + expected) or expected.endswith("." + host):
                return candidate
    return found[0]


def _decode_google_news_url(raw: str, expected_domain: str = "") -> str | None:
    article_id = _google_news_article_id(raw)
    if not article_id:
        return None
    legacy = _legacy_google_news_decode(article_id)
    if legacy:
        return legacy
    params = _fetch_google_params(article_id)
    if not params:
        return None
    aid, signature, timestamp = params
    inner = [
        "garturlreq",
        [
            ["X", "X", ["X", "X"], None, None, 1, 1, "US:en", None, 1, None, None, None, None, None, 0, 1],
            "X", "X", 1, [1, 1, 1], 1, 1, None, 0, 0, None, 0,
        ],
        aid,
        int(timestamp),
        signature,
    ]
    rpc = ["Fbv4je", json.dumps(inner, separators=(",", ":"), ensure_ascii=False)]
    f_req = json.dumps([[rpc]], separators=(",", ":"), ensure_ascii=False)
    payload = urllib.parse.urlencode({"f.req": f_req}).encode("utf-8")
    req = urllib.request.Request(
        "https://news.google.com/_/DotsSplashUi/data/batchexecute",
        data=payload,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/123 Safari/537.36",
            "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
            "Accept-Language": "en-US,en;q=0.8",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            text = resp.read(3 * 1024 * 1024).decode(
                resp.headers.get_content_charset() or "utf-8", errors="replace"
            )
    except Exception:
        return None

    decoder = json.JSONDecoder()
    for start, ch in enumerate(text):
        if ch != "[":
            continue
        try:
            obj, _ = decoder.raw_decode(text[start:])
        except Exception:
            continue
        candidate = _find_external_url(obj, expected_domain)
        if candidate:
            return candidate
    return None


def _save_source_cache() -> None:
    try:
        SOURCE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        temp = SOURCE_CACHE_PATH.with_suffix(".tmp")
        temp.write_text(json.dumps(SOURCE_CACHE, indent=2, ensure_ascii=False), encoding="utf-8")
        temp.replace(SOURCE_CACHE_PATH)
    except Exception:
        pass


def _resolve_source_url(raw: str, expected_domain: str = "") -> tuple[str, bool]:
    if not raw:
        return raw, False
    if not _is_google_news_url(raw):
        return raw, False
    with SOURCE_CACHE_LOCK:
        cached = SOURCE_CACHE.get(raw)
    if cached:
        return cached, True
    decoded = _decode_google_news_url(raw, expected_domain)
    if decoded:
        try:
            _public_http_url(decoded)
        except Exception:
            decoded = None
    if decoded and not _is_google_news_url(decoded):
        with SOURCE_CACHE_LOCK:
            SOURCE_CACHE[raw] = decoded
            _save_source_cache()
        return decoded, True
    return raw, False


class _ReadableHTMLParser(HTMLParser):
    BLOCK_TAGS = {"p", "div", "section", "article", "main", "li", "ul", "ol", "blockquote", "pre", "h1", "h2", "h3", "h4", "h5", "h6", "br", "tr", "table"}
    SKIP_TAGS = {"script", "style", "noscript", "svg", "canvas", "iframe", "form", "button", "input", "header", "footer", "nav", "aside", "figure", "figcaption"}

    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip_depth = 0
        self.title = ""
        self._capture_title = False

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == "title":
            self._capture_title = True
        if tag in self.SKIP_TAGS:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag == "title":
            self._capture_title = False
        if tag in self.SKIP_TAGS and self.skip_depth:
            self.skip_depth -= 1
            return
        if self.skip_depth:
            return
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._capture_title:
            self.title += data
        if self.skip_depth:
            return
        text = re.sub(r"\s+", " ", data or "").strip()
        if text:
            self.parts.append(text + " ")


def _safe_text_fragment(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(text or "")).strip()


def _highlight_html(text: str, target: str) -> str:
    escaped = html.escape(text or "")
    target = _safe_text_fragment(target)
    if not escaped or not target:
        return escaped
    variants = [target, target[:180].rsplit(" ", 1)[0], " ".join(target.split(" ")[:16])]
    out = escaped
    for variant in variants:
        needle = html.escape(variant.strip())
        if len(needle) < 12:
            continue
        pattern = re.compile(re.escape(needle), re.I)
        if pattern.search(out):
            out = pattern.sub(lambda m: f'<mark data-schain-hit="1">{m.group(0)}</mark>', out, count=1)
            break
    return out


def _build_reader_preview(document: str, source_url: str, target_text: str) -> str | None:
    parser = _ReadableHTMLParser()
    try:
        parser.feed(document)
    except Exception:
        return None

    title = parser.title.strip()
    og_title = re.search(r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)', document, re.I)
    meta_desc = re.search(r'<meta[^>]+(?:name=["\']description["\']|property=["\']og:description["\'])[^>]+content=["\']([^"\']+)', document, re.I)
    if og_title and len(og_title.group(1).strip()) > len(title):
        title = html.unescape(og_title.group(1).strip())
    description = html.unescape(meta_desc.group(1).strip()) if meta_desc else ""

    text = _safe_text_fragment(" ".join(parser.parts))
    if not text:
        return None
    chunks = [c.strip() for c in re.split(r'(?<=[\.!?])\s+(?=[A-Z0-9"“])', text) if c.strip()]
    paragraphs, buf, seen = [], "", set()
    for chunk in chunks:
        if len(chunk) < 35:
            continue
        low = chunk.lower()
        if low in seen:
            continue
        seen.add(low)
        if len(buf) + len(chunk) < 520:
            buf = (buf + " " + chunk).strip()
        else:
            if buf:
                paragraphs.append(buf)
            buf = chunk
        if len(paragraphs) >= 18:
            break
    if buf and len(paragraphs) < 18:
        paragraphs.append(buf)
    if description and all(description[:60].lower() not in p.lower() for p in paragraphs[:3]):
        paragraphs.insert(0, description)
    if len(paragraphs) < 2:
        return None

    paras_html = "".join(f"<p>{_highlight_html(p, target_text)}</p>" for p in paragraphs)
    src_display = html.escape(source_url)
    title_html = html.escape(title or source_url)
    domain = html.escape(urllib.parse.urlsplit(source_url).hostname or "")
    return f"""<!doctype html><html><head><meta charset='utf-8'><style>
    body{{margin:0;background:#f5f7fb;color:#12202f;font:16px/1.7 Inter,system-ui,sans-serif;}}
    .wrap{{max-width:860px;margin:0 auto;padding:28px 28px 56px;}}
    .card{{background:#fff;border:1px solid #dde5f1;border-radius:18px;box-shadow:0 12px 30px rgba(16,24,40,.08);overflow:hidden;}}
    .head{{padding:22px 24px 16px;border-bottom:1px solid #edf2f7;background:linear-gradient(180deg,#fbfcfe,#f7f9fc);}}
    .domain{{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:#678;}}
    h1{{margin:8px 0 0;font-size:28px;line-height:1.25;color:#101828;}}
    .meta{{margin-top:10px;font-size:13px;color:#667085;word-break:break-all;}}
    .content{{padding:24px;}}
    p{{margin:0 0 16px;}}
    mark[data-schain-hit]{{background:#ffe36e;color:#111;padding:0 .08em;border-radius:.14em;}}
    .note{{font-size:12px;color:#8090a6;margin-top:18px;}}
    </style></head><body><div class='wrap'><div class='card'><div class='head'><div class='domain'>{domain}</div><h1>{title_html}</h1><div class='meta'>{src_display}</div></div><div class='content'>{paras_html}<div class='note'>Reader preview is used here so evidence can scroll and highlight reliably even when the publisher page does not embed well.</div></div></div></div><script>(function(){{var el=document.querySelector('mark[data-schain-hit]');if(el){{setTimeout(function(){{el.scrollIntoView({{block:'center',behavior:'instant'}});}},60);}}}})();</script></body></html>"""


def _save_evidence_content_cache() -> None:
    try:
        EVIDENCE_CONTENT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = EVIDENCE_CONTENT_CACHE_PATH.with_suffix('.tmp')
        tmp.write_text(json.dumps(EVIDENCE_CONTENT_CACHE, indent=2, ensure_ascii=False), encoding='utf-8')
        tmp.replace(EVIDENCE_CONTENT_CACHE_PATH)
    except Exception:
        pass


def _content_cache_key(url: str, quote: str) -> str:
    raw = (url or '') + '\n' + _safe_text_fragment(quote or '')[:600]
    return hashlib.sha256(raw.encode('utf-8', errors='ignore')).hexdigest()[:32]


def _extract_jsonld_article(document: str) -> tuple[str, str]:
    title = ''
    bodies = []
    for m in re.finditer(r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', document, re.I | re.S):
        blob = html.unescape(m.group(1)).strip()
        try:
            data = json.loads(blob)
        except Exception:
            continue
        stack = [data]
        while stack:
            obj = stack.pop()
            if isinstance(obj, list):
                stack.extend(obj)
                continue
            if not isinstance(obj, dict):
                continue
            typ = obj.get('@type')
            types = typ if isinstance(typ, list) else [typ]
            is_article = any(str(x).lower() in {'article','newsarticle','report','blogposting','analysisnewsarticle'} for x in types if x)
            body = obj.get('articleBody') or obj.get('text')
            if is_article and isinstance(body, str) and len(body.strip()) > 180:
                bodies.append(_safe_text_fragment(body))
            if not title:
                t = obj.get('headline') or obj.get('name')
                if isinstance(t, str):
                    title = _safe_text_fragment(t)
            for v in obj.values():
                if isinstance(v, (dict, list)):
                    stack.append(v)
    body = max(bodies, key=len) if bodies else ''
    return title, body


def _extract_amp_url(document: str, base_url: str) -> str | None:
    m = re.search(r'<link[^>]+rel=["\']amphtml["\'][^>]+href=["\']([^"\']+)', document, re.I)
    if not m:
        m = re.search(r'<link[^>]+href=["\']([^"\']+)["\'][^>]+rel=["\']amphtml["\']', document, re.I)
    if not m:
        return None
    return urllib.parse.urljoin(base_url, html.unescape(m.group(1)))


def _normalize_for_match(value: str) -> str:
    return re.sub(r'[^a-z0-9]+', ' ', _safe_text_fragment(value).lower()).strip()


def _context_window(body: str, quote: str, max_chars: int = 3200) -> tuple[str, bool]:
    body = _safe_text_fragment(body)
    quote = _safe_text_fragment(quote)
    if not body:
        return '', False
    if len(body) <= max_chars:
        return body, bool(quote and _normalize_for_match(quote) in _normalize_for_match(body))
    lower = body.lower()
    q = quote.lower().strip()
    idx = lower.find(q) if q else -1
    if idx < 0 and quote:
        short = ' '.join(quote.split()[:14]).lower()
        idx = lower.find(short) if len(short) > 20 else -1
    if idx < 0:

        tokens = [t for t in re.findall(r'[A-Za-z0-9][A-Za-z0-9\-\']+', quote) if len(t) > 4]
        for start in range(max(0, len(tokens)-5)):
            piece = ' '.join(tokens[start:start+5]).lower()
            idx = lower.find(piece)
            if idx >= 0:
                break
    matched = idx >= 0
    if idx < 0:
        return body[:max_chars].rsplit(' ', 1)[0] + '…', False
    center = idx + max(1, len(quote)) // 2
    start = max(0, center - max_chars // 2)
    end = min(len(body), start + max_chars)
    start = max(0, body.rfind(' ', 0, start))
    if end < len(body):
        end = body.find(' ', end)
        if end < 0:
            end = len(body)
    snippet = body[start:end].strip()
    if start > 0:
        snippet = '…' + snippet
    if end < len(body):
        snippet += '…'
    return snippet, matched


def _paragraphize(text: str) -> list[str]:
    text = _safe_text_fragment(text)
    if not text:
        return []
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"“])', text)
    out, buf = [], ''
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(buf) + len(s) + 1 <= 700:
            buf = (buf + ' ' + s).strip()
        else:
            if buf:
                out.append(buf)
            buf = s
    if buf:
        out.append(buf)
    return out[:8]


def _fetch_html(url: str) -> tuple[str, str]:
    req = urllib.request.Request(
        _public_http_url(url),
        headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/123 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.8',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache',
        },
    )
    with urllib.request.urlopen(req, timeout=14) as resp:
        ctype = resp.headers.get('Content-Type', '')
        if 'text/html' not in ctype and 'application/xhtml+xml' not in ctype:
            raise ValueError('Source is not an HTML page.')
        raw = resp.read(MAX_PREVIEW_BYTES + 1)
        if len(raw) > MAX_PREVIEW_BYTES:
            raise ValueError('Source page is too large.')
        charset = resp.headers.get_content_charset() or 'utf-8'
        return raw.decode(charset, errors='replace'), resp.geturl()


def _extract_readable_text(document: str) -> tuple[str, str]:
    jsonld_title, jsonld_body = _extract_jsonld_article(document)
    if jsonld_body:
        return jsonld_title, jsonld_body
    parser = _ReadableHTMLParser()
    try:
        parser.feed(document)
    except Exception:
        pass
    title = jsonld_title or parser.title.strip()
    text = _safe_text_fragment(' '.join(parser.parts))

    text = re.sub(r'(accept all cookies|privacy policy|terms of use|sign in|subscribe)\s+', ' ', text, flags=re.I)
    return title, text


def _title_similarity(a: str, b: str) -> float:
    aa = set(re.findall(r'[a-z0-9]+', _safe_text_fragment(a).lower()))
    bb = set(re.findall(r'[a-z0-9]+', _safe_text_fragment(b).lower()))
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / max(1, len(aa | bb))


def _quote_match_score(body: str, quote: str) -> tuple[bool, float]:
    nb = _normalize_for_match(body)
    nq = _normalize_for_match(quote)
    if not nq or not nb:
        return False, 0.0
    if nq in nb:
        return True, 1.0
    tokens = [t for t in nq.split() if len(t) > 3]
    if not tokens:
        return False, 0.0
    best = 0
    for width in (12, 10, 8, 6, 5):
        if len(tokens) < width:
            continue
        for i in range(0, min(len(tokens) - width + 1, 18)):
            piece = ' '.join(tokens[i:i+width])
            if piece in nb:
                best = max(best, width / max(12, len(tokens)))
                break
        if best:
            break
    return best >= 0.25, best


def _google_news_search_candidates(title: str, quote: str, limit: int = 8) -> list[tuple[str, str]]:
    query = _safe_text_fragment(title)
    if len(query) < 20:
        query = ' '.join(_safe_text_fragment(quote).split()[:18])
    if not query:
        return []

    q = f'"{query[:240]}"'
    url = 'https://news.google.com/rss/search?' + urllib.parse.urlencode({
        'q': q, 'hl': 'en-US', 'gl': 'US', 'ceid': 'US:en'
    })
    req = urllib.request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/123 Safari/537.36',
        'Accept': 'application/rss+xml,application/xml,text/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.8',
    })
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read(1024 * 1024)
        root = ET.fromstring(raw)
    except Exception:
        return []
    out = []
    for item in root.findall('.//item')[:limit * 2]:
        link = _safe_text_fragment(item.findtext('link') or '')
        ititle = _safe_text_fragment(item.findtext('title') or '')
        if not link:
            continue
        try:
            if _is_google_news_url(link):
                direct, _ = _resolve_source_url(link)
            else:
                direct = link
            if direct and not _is_google_news_url(direct):
                out.append((direct, ititle))
        except Exception:
            continue
        if len(out) >= limit:
            break
    return out


def _decode_ddg_href(href: str) -> str:
    href = html.unescape(href or '')
    if href.startswith('//'):
        href = 'https:' + href
    try:
        p = urllib.parse.urlsplit(href)
        if 'duckduckgo.com' in (p.hostname or ''):
            qs = urllib.parse.parse_qs(p.query)
            target = (qs.get('uddg') or [''])[0]
            if target:
                return urllib.parse.unquote(target)
    except Exception:
        pass
    return href


def _duckduckgo_search_candidates(title: str, quote: str, limit: int = 8) -> list[tuple[str, str]]:
    pieces = []
    if title:
        pieces.append(f'"{_safe_text_fragment(title)[:220]}"')
    qfrag = ' '.join(_safe_text_fragment(quote).split()[:12])
    if qfrag:
        pieces.append(f'"{qfrag}"')
    query = ' '.join(pieces).strip()
    if not query:
        return []
    url = 'https://html.duckduckgo.com/html/?' + urllib.parse.urlencode({'q': query})
    req = urllib.request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/123 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.8',
    })
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            document = resp.read(2 * 1024 * 1024).decode(resp.headers.get_content_charset() or 'utf-8', errors='replace')
    except Exception:
        return []
    out = []
    for m in re.finditer(r'<a[^>]+class=["\'][^"\']*result__a[^"\']*["\'][^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', document, re.I | re.S):
        target = _decode_ddg_href(m.group(1))
        label = _safe_text_fragment(re.sub(r'<[^>]+>', ' ', m.group(2)))
        try:
            host = (urllib.parse.urlsplit(target).hostname or '').lower()
            if target.startswith(('http://', 'https://')) and host and 'duckduckgo.com' not in host:
                out.append((target, label))
        except Exception:
            pass
        if len(out) >= limit:
            break
    return out


def _search_public_article(title: str, quote: str, expected_domain: str = '', original_url: str = '') -> dict | None:
    candidates = []
    seen = set()
    for url, label in _google_news_search_candidates(title, quote, 8) + _duckduckgo_search_candidates(title, quote, 8):
        try:
            norm = urllib.parse.urlunsplit(urllib.parse.urlsplit(url)._replace(fragment=''))
        except Exception:
            norm = url
        if not norm or norm in seen or norm == original_url:
            continue
        seen.add(norm)
        candidates.append((norm, label))
        if len(candidates) >= 10:
            break

    best = None
    for url, label in candidates[:6]:
        try:
            document, final_url = _fetch_html(url)
            page_title, body = _extract_readable_text(document)
            if len(body) < 700:
                amp = _extract_amp_url(document, final_url)
                if amp:
                    try:
                        amp_doc, amp_final = _fetch_html(amp)
                        a_title, a_body = _extract_readable_text(amp_doc)
                        if len(a_body) > len(body):
                            page_title, body, final_url = a_title or page_title, a_body, amp_final
                    except Exception:
                        pass
            if len(body) < 650:
                continue
            matched, qscore = _quote_match_score(body, quote)
            tsim = max(_title_similarity(title, page_title), _title_similarity(title, label))
            domain = urllib.parse.urlsplit(final_url).hostname or ''
            same_domain = bool(expected_domain and expected_domain.lower().replace('www.','') in domain.lower().replace('www.',''))
            score = (120 if matched else 0) + qscore * 60 + tsim * 45 + (15 if same_domain else 0) + min(len(body), 12000) / 1200

            if not matched and tsim < 0.46:
                continue
            row = {'title': page_title or label or title, 'body': body, 'url': final_url, 'domain': domain, 'matched': matched, 'score': score}
            if not best or score > best['score']:
                best = row

            if matched and len(body) > 1800 and tsim > 0.35:
                break
        except Exception:
            continue
    return best


def _paragraphize_full(text: str, max_paragraphs: int = 70) -> list[str]:
    text = _safe_text_fragment(text)
    if not text:
        return []
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"“])', text)
    out, buf = [], ''
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(buf) + len(s) + 1 <= 950:
            buf = (buf + ' ' + s).strip()
        else:
            if buf:
                out.append(buf)
            buf = s
        if len(out) >= max_paragraphs:
            break
    if buf and len(out) < max_paragraphs:
        out.append(buf)
    return out


def _content_from_site(raw_url: str, quote: str, expected_domain: str = '', search_title: str = '') -> dict:
    resolved, _ = _resolve_source_url(raw_url, expected_domain) if _is_google_news_url(raw_url) else (raw_url, False)
    direct_error = None
    direct = None
    try:
        document, final_url = _fetch_html(resolved)
        title, body = _extract_readable_text(document)
        if len(body) < 700:
            amp = _extract_amp_url(document, final_url)
            if amp:
                try:
                    amp_doc, amp_final = _fetch_html(amp)
                    a_title, a_body = _extract_readable_text(amp_doc)
                    if len(a_body) > len(body):
                        title, body, final_url = a_title or title, a_body, amp_final
                except Exception:
                    pass
        matched, _ = _quote_match_score(body, quote)
        direct = {'title': title or search_title, 'body': body, 'url': final_url, 'domain': urllib.parse.urlsplit(final_url).hostname or expected_domain, 'matched': matched}
    except Exception as exc:
        direct_error = str(exc)


    best = direct if direct and len(direct.get('body','')) >= 1400 else None
    should_search = not best or not best.get('matched') or len(best.get('body','')) < 2200
    if should_search:
        searched = _search_public_article(search_title or (direct or {}).get('title',''), quote, expected_domain, resolved)
        if searched and (not best or searched.get('score',0) >= 105 or len(searched.get('body','')) > len(best.get('body','')) * 1.35):
            best = searched
            origin = 'web-search'
        else:
            origin = 'web'
    else:
        origin = 'web'

    if not best or not best.get('body'):
        if direct and direct.get('body'):
            best = direct
            origin = 'web'
        else:
            raise ValueError(direct_error or 'Readable source content could not be extracted or found publicly.')

    body = _safe_text_fragment(best.get('body',''))


    paragraphs = _paragraphize_full(body)
    if not paragraphs:
        context, matched = _context_window(body, quote, max_chars=5200)
        paragraphs = _paragraphize(context)
    else:
        matched = bool(best.get('matched'))
    return {
        'title': best.get('title') or search_title,
        'domain': best.get('domain') or expected_domain,
        'url': best.get('url') or resolved,
        'context': body,
        'paragraphs': paragraphs,
        'matched': bool(matched),
        'origin': origin,
        'full_text': True,
        'searched': origin == 'web-search',
    }


def _inject_preview_helpers(document: str, base_url: str, target_text: str) -> str:
    base_tag = f'<base href="{html.escape(base_url, quote=True)}">'
    helper = f"""
<style id="schain-preview-style">
  ::selection {{ background: #ffe36e; color: #111; }}
  mark[data-schain-hit] {{ background:#ffe36e!important; color:#111!important; padding:0 .06em; border-radius:.12em; }}
</style>
<script id="schain-preview-script">
(() => {{
  const target = {json.dumps(target_text, ensure_ascii=False)}.replace(/\\s+/g,' ').trim();
  if (!target) return;
  const esc = s => s.replace(/[.*+?^${{}}()|[\\]\\]/g,'\\$&');
  const tryMark = () => {{
    const needles = [target, target.slice(0,180), target.split(' ').slice(0,16).join(' ')].filter(x => x.length > 12);
    const walker = document.createTreeWalker(document.body || document.documentElement, NodeFilter.SHOW_TEXT);
    for (const needle of needles) {{
      const lower = needle.toLowerCase();
      walker.currentNode = document.body || document.documentElement;
      let node;
      while ((node = walker.nextNode())) {{
        const value = (node.nodeValue || '');
        const idx = value.toLowerCase().indexOf(lower);
        if (idx >= 0) {{
          const range = document.createRange();
          range.setStart(node, idx); range.setEnd(node, idx + needle.length);
          const mark = document.createElement('mark'); mark.dataset.schainHit = '1';
          try {{ range.surroundContents(mark); mark.scrollIntoView({{block:'center', behavior:'instant'}}); return true; }} catch (_) {{}}
        }}
      }}
    }}
    for (const needle of needles) {{
      try {{ if (window.find && window.find(needle, false, false, true, false, true, false)) return true; }} catch (_) {{}}
    }}
    return false;
  }};
  const run = () => {{ tryMark(); setTimeout(tryMark, 700); }};
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', run, {{once:true}}); else run();
}})();
</script>
"""
    lower = document.lower()

    document = re.sub(r'<meta[^>]+http-equiv=["\']?content-security-policy["\']?[^>]*>', '', document, flags=re.I)
    lower = document.lower()
    if "<head" in lower:
        pos = lower.find('>', lower.find('<head'))
        if pos >= 0:
            return document[:pos+1] + base_tag + helper + document[pos+1:]
    return '<!doctype html><html><head>' + base_tag + helper + '</head><body>' + document + '</body></html>'


class QuietHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, fmt, *args):
        pass

    def _send_json(self, data: dict) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path == "/resolve-source":
            self._serve_source_resolution(parsed)
            return
        if parsed.path == "/evidence-preview":
            self._serve_evidence_preview(parsed)
            return
        if parsed.path == "/evidence-content":
            self._serve_evidence_content(parsed)
            return
        super().do_GET()

    def _serve_source_resolution(self, parsed):
        params = urllib.parse.parse_qs(parsed.query)
        raw_url = (params.get("url") or [""])[0]
        expected_domain = (params.get("publisher") or [""])[0][:255]
        try:
            if _is_google_news_url(raw_url):
                resolved, changed = _resolve_source_url(raw_url, expected_domain)
            else:
                resolved = _public_http_url(raw_url)
                changed = False
            result = {"url": resolved, "resolved": changed}
        except Exception as exc:
            result = {"url": raw_url, "resolved": False, "error": str(exc)}
        self._send_json(result)

    def _serve_evidence_content(self, parsed):
        params = urllib.parse.parse_qs(parsed.query)
        raw_url = (params.get('url') or [''])[0]
        quote = (params.get('text') or [''])[0][:800]
        expected_domain = (params.get('publisher') or [''])[0][:255]
        fallback_title = (params.get('title') or [''])[0][:500]
        key = _content_cache_key(raw_url, quote)
        with EVIDENCE_CONTENT_CACHE_LOCK:
            cached = EVIDENCE_CONTENT_CACHE.get(key)
        if cached:
            self._send_json(cached)
            return
        try:
            result = _content_from_site(raw_url, quote, expected_domain, fallback_title)
            if not result.get('title'):
                result['title'] = fallback_title
            result['ok'] = True
            with EVIDENCE_CONTENT_CACHE_LOCK:
                EVIDENCE_CONTENT_CACHE[key] = result
                _save_evidence_content_cache()
        except Exception as exc:
            result = {
                'ok': False,
                'title': fallback_title,
                'domain': expected_domain or (urllib.parse.urlsplit(raw_url).hostname or ''),
                'url': raw_url,
                'context': quote,
                'paragraphs': [quote] if quote else [],
                'matched': False,
                'origin': 'evidence',
                'error': str(exc),
            }
        self._send_json(result)

    def _serve_evidence_preview(self, parsed):
        params = urllib.parse.parse_qs(parsed.query)
        raw_url = (params.get("url") or [""])[0]
        target = (params.get("text") or [""])[0][:600]
        try:
            resolved_url, _ = _resolve_source_url(raw_url) if _is_google_news_url(raw_url) else (raw_url, False)
            url = _public_http_url(resolved_url)
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/123 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.8",
                    "Cache-Control": "no-cache",
                    "Pragma": "no-cache",
                    "Referer": url,
                },
            )
            with urllib.request.urlopen(req, timeout=12) as resp:
                content_type = resp.headers.get("Content-Type", "")
                if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
                    raise ValueError("The source is not an HTML page.")
                raw = resp.read(MAX_PREVIEW_BYTES + 1)
                if len(raw) > MAX_PREVIEW_BYTES:
                    raise ValueError("The source page is too large to preview.")
                charset = resp.headers.get_content_charset() or "utf-8"
                document = raw.decode(charset, errors="replace")
                final_url = resp.geturl()
            reader_doc = _build_reader_preview(document, final_url, target)
            if reader_doc:
                document = reader_doc
            else:
                document = _inject_preview_helpers(document, final_url, target)
            body = document.encode("utf-8", errors="replace")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            message = html.escape(str(exc) or "Preview unavailable")
            external = html.escape(raw_url, quote=True)
            body = f"""<!doctype html><html><head><meta charset='utf-8'><style>
            body{{margin:0;background:#fff;color:#1c2530;font:15px/1.5 system-ui;padding:40px}}
            .box{{max-width:720px;margin:10vh auto;border:1px solid #dfe4ea;border-radius:14px;padding:24px}}
            a{{color:#315fe8}}</style></head><body><div class='box'><b>Preview unavailable</b><p>{message}</p><a href='{external}' target='_blank' rel='noopener'>Open source</a></div></body></html>""".encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    url = f"http://127.0.0.1:{args.port}/"
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), QuietHandler)
    print(f"SChain Graph Explorer: {url}")
    print("Press Ctrl+C to stop.")

    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
