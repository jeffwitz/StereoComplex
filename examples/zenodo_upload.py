#!/usr/bin/env python3
"""Create and upload a new version of a Zenodo record (CMO paper artefacts).

Drives the Zenodo REST API to deposit a *new version* of an existing record:
it creates the draft, optionally clears the inherited files, uploads the given
files through the bucket API, optionally updates the version string / date, and
prints the reserved DOI and the draft URL. By default it **stops before
publishing** — review the draft in the browser, then re-run with ``--publish``
(or click *Publish* on Zenodo). Publishing mints a DOI and is irreversible.

Two records back the CMO paper:

* the self-contained **paper archive** (concept DOI 10.5281/zenodo.20444215);
  upload the structured bundle from ``paper/cmo/make_zenodo_bundle.py`` plus the
  manuscript PDF::

      ZENODO_TOKEN=... python examples/zenodo_upload.py --record <latest_version_id> \\
          --replace --version v5 \\
          --files paper/cmo/build/cmo_paper_bundle.zip paper/cmo/manuscript.pdf

* the heavy **specimen reconstructions** (10.5281/zenodo.20369312); upload the
  regenerated full-image ``.npz``::

      ZENODO_TOKEN=... python examples/zenodo_upload.py --record 20369312 --replace \\
          --files docs/assets/pycaso_real_data/schur_ba/specimen_*.npz

Pass ``--record`` the numeric id of the **latest published version** (not the
concept-DOI id). Use ``--sandbox`` against https://sandbox.zenodo.org first.
``ZENODO_TOKEN`` (env) or ``--token`` must carry a token with the
``deposit:write`` (and ``deposit:actions`` for ``--publish``) scopes.

This script performs outward-facing writes; nothing is sent under ``--dry-run``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def _api(method: str, url: str, token: str, *, data: bytes | None = None,
         json_body: dict | None = None, ctype: str | None = None) -> dict | None:
    """Issue one authenticated Zenodo API call; return the parsed JSON (or None)."""
    headers = {"Authorization": f"Bearer {token}"}
    body = data
    if json_body is not None:
        body = json.dumps(json_body).encode()
        headers["Content-Type"] = "application/json"
    elif ctype is not None:
        headers["Content-Type"] = ctype
    req = Request(url, data=body, method=method, headers=headers)
    try:
        with urlopen(req, timeout=600) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else None
    except HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise SystemExit(f"Zenodo API {method} {url} -> {exc.code}\n{detail}") from exc


def new_version_draft(base: str, record_id: str, token: str) -> dict:
    """Create (or reuse) the new-version draft of a published record."""
    dep = _api("POST", f"{base}/api/deposit/depositions/{record_id}/actions/newversion",
               token)
    draft_url = dep["links"]["latest_draft"]
    return _api("GET", draft_url, token)


def clear_files(base: str, draft_id: int, token: str) -> None:
    """Delete every file inherited by the draft (so uploads fully replace them)."""
    files = _api("GET", f"{base}/api/deposit/depositions/{draft_id}/files", token) or []
    for f in files:
        _api("DELETE", f"{base}/api/deposit/depositions/{draft_id}/files/{f['id']}", token)
        print(f"  deleted inherited file: {f['filename']}")


def upload(bucket_url: str, path: Path, token: str) -> None:
    """Stream one file to the draft's bucket (Zenodo new-style files API)."""
    with path.open("rb") as fh:
        _api("PUT", f"{bucket_url}/{path.name}", token, data=fh.read(),
             ctype="application/octet-stream")
    print(f"  uploaded {path.name} ({path.stat().st_size / 1e6:.1f} MB)")


def set_metadata(base: str, draft: dict, token: str,
                 version: str | None, pub_date: str | None) -> None:
    """Patch the draft's version string / publication date in place."""
    meta = dict(draft.get("metadata", {}))
    if version:
        meta["version"] = version
    if pub_date:
        meta["publication_date"] = pub_date
    _api("PUT", f"{base}/api/deposit/depositions/{draft['id']}", token,
         json_body={"metadata": meta})
    print(f"  metadata updated (version={meta.get('version')}, "
          f"date={meta.get('publication_date')})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", required=True,
                    help="numeric id of the latest PUBLISHED version of the record")
    ap.add_argument("--files", nargs="+", required=True, type=Path,
                    help="files to upload into the new version")
    ap.add_argument("--replace", action="store_true",
                    help="delete the files inherited from the previous version first")
    ap.add_argument("--version", help="version string for the new deposit (e.g. v5)")
    ap.add_argument("--publication-date", help="ISO date YYYY-MM-DD for the new version")
    ap.add_argument("--sandbox", action="store_true",
                    help="use https://sandbox.zenodo.org instead of production")
    ap.add_argument("--token", default=os.environ.get("ZENODO_TOKEN"),
                    help="API token (defaults to $ZENODO_TOKEN)")
    ap.add_argument("--publish", action="store_true",
                    help="publish the draft after upload (IRREVERSIBLE — mints the DOI)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the planned actions and exit without contacting Zenodo")
    args = ap.parse_args()

    missing = [str(p) for p in args.files if not p.is_file()]
    if missing:
        raise SystemExit(f"files not found: {', '.join(missing)}")
    base = "https://sandbox.zenodo.org" if args.sandbox else "https://zenodo.org"

    if args.dry_run:
        print(f"[dry-run] target: {base}  record: {args.record}")
        print(f"[dry-run] replace inherited files: {args.replace}")
        print(f"[dry-run] version={args.version} date={args.publication_date}")
        for p in args.files:
            print(f"[dry-run] would upload {p} ({p.stat().st_size / 1e6:.1f} MB)")
        print(f"[dry-run] publish: {args.publish}")
        return 0

    if not args.token:
        raise SystemExit("no token — set ZENODO_TOKEN or pass --token")

    print(f"==> new version of record {args.record} on {base}")
    draft = new_version_draft(base, args.record, args.token)
    draft_id = draft["id"]
    bucket = draft["links"]["bucket"]
    print(f"  draft deposition id: {draft_id}")

    if args.replace:
        clear_files(base, draft_id, args.token)
    for p in args.files:
        upload(bucket, p, args.token)
    if args.version or args.publication_date:
        set_metadata(base, draft, args.token, args.version, args.publication_date)

    reserved = (draft.get("metadata", {}).get("prereserve_doi") or {}).get("doi")
    html = draft["links"].get("html", f"{base}/deposit/{draft_id}")
    print(f"  reserved DOI: {reserved}")
    print(f"  review the draft: {html}")

    if args.publish:
        pub = _api("POST", f"{base}/api/deposit/depositions/{draft_id}/actions/publish",
                   args.token)
        print(f"  PUBLISHED: DOI {pub['doi']}  ({pub['links']['record_html']})")
    else:
        print("  not published — review, then re-run with --publish or click Publish.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
