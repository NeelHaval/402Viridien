"""Check three request IDs with public metadata queries. Run with Python -I.

No login, request-cart, email, or file-request action is used.
"""

import csv
import datetime as dt
import hashlib
import html
from html.parser import HTMLParser
import http.cookiejar
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request


ROOT = Path(__file__).resolve().parents[3]
NOTES = ROOT / "bakeoff" / "notes" / "bsee_list"
SOURCES = ROOT / "external_untrusted" / "bsee" / "sources"
URL = "https://www.data.bsee.gov/Other/FileRequestSystem/WellData.aspx"
PREFIX = "ctl00$ContentPlaceHolder1$ASPxFormLayout1$"


class Inputs(HTMLParser):
    def __init__(self):
        super().__init__()
        self.values = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "input" and attrs.get("name") and attrs.get("type", "text") not in ("submit", "button", "checkbox", "radio", "file"):
            self.values[attrs["name"]] = attrs.get("value", "")


def main():
    with (NOTES / "pilot_request.csv").open(newline="", encoding="utf-8") as stream:
        docs = list(csv.DictReader(stream))
    # Check one leading generic log and two with stronger content terms.
    sample = [next(d for d in docs if d["DOC_ID"] == doc_id) for doc_id in ("clog06009", "clog16345", "clog11745")]
    checks = []
    for doc in sample:
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        opener.addheaders = [("User-Agent", "Mozilla/5.0")]
        with opener.open(URL, timeout=45) as response:
            parser = Inputs()
            parser.feed(response.read(2_000_000).decode())
        form = parser.values
        # A search POST returns public metadata. It does not submit a file request.
        form["__EVENTTARGET"] = PREFIX + "ASPxButtonSubmitQ"
        form["__EVENTARGUMENT"] = ""
        form[PREFIX + "ASPxTextBoxID"] = doc["DOC_ID"]
        assert not any("@" in v for v in form.values()), "Do not send an email address"
        with opener.open(URL, urllib.parse.urlencode(form).encode(), timeout=60) as response:
            assert response.url == URL
            body = response.read(2_000_000)
        path = SOURCES / f"portal_{doc['DOC_ID']}.html"
        path.write_bytes(body)
        rows = []
        for row in re.findall(r"<tr[^>]*DXDataRow[^>]*>(.*?)</tr>", body.decode(), re.S):
            cells = re.findall(r"<td\b[^>]*>(.*?)</td>", row, re.S)
            cells = [re.sub(r"\s+", " ", html.unescape(re.sub("<[^>]+>", " ", cell))).strip() for cell in cells]
            # The first and last cells hold grid controls. Compare the 15
            # data cells between them, with no change to the field text.
            assert not cells[0] and all(not cell for cell in cells[16:])
            rows.append(cells[1:16])
        expected = [doc["API"], doc["WELL_NAME"].strip(), doc["WELL_NAME_SUFFIX"], doc["LEASE"].strip(), doc["AREA"], doc["BLOCK"].strip(), re.sub(r"\s+", " ", doc["DOC_TYPE"]).strip(), doc["RUN_DATE"], doc["RELEASABLE_DATE"], doc["BLI"], doc["TLI"], doc["WATER_DEPTH"], doc["REGION"], doc["CREATED_DATE"], doc["FILE_SIZE"]]
        assert rows == [expected], (doc["DOC_ID"], rows, expected)
        checks.append({
            "doc_id": doc["DOC_ID"], "url": URL,
            "public_query_field": "Filename ID", "query_value": doc["DOC_ID"],
            "checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "saved_response": path.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(body).hexdigest(),
            "result": "PASS: one portal row matches all 15 displayed inventory fields",
            "cells": rows[0],
        })
        print(doc["DOC_ID"], "PASS")
    (NOTES / "portal_checks.json").write_text(json.dumps(checks, indent=2) + "\n")


if __name__ == "__main__":
    main()
