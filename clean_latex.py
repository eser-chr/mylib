import re
import sys
from pathlib import Path


def get_cite_keys(tex_path):
    text = Path(tex_path).read_text()
    text = re.sub(r"(?<!\\)%.*", "", text)  # drop comments
    groups = re.findall(r"\\\w*cite\w*\*?(?:\[[^\]]*\])*\{([^}]*)\}", text)
    keys = [k.strip() for g in groups for k in g.split(",") if k.strip()]
    return list(dict.fromkeys(keys))  # unique, keeps order


def filter_bib(keys, bib_path):
    text = Path(bib_path).read_text()
    entries = re.split(r"(?m)^(?=@)", text)
    keys = set(keys)
    kept = []
    for entry in entries:
        m = re.match(r"@\w+\s*\{\s*([^,\s]+)", entry)
        if m and m.group(1) in keys:
            kept.append(entry.strip())
    return kept


if __name__ == "__main__":
    tex, bib, out = sys.argv[1], sys.argv[2], sys.argv[3]
    keys = get_cite_keys(tex)
    kept = filter_bib(keys, bib)
    Path(out).write_text("\n\n".join(kept) + "\n")
    print(f"{len(keys)} cited, {len(kept)} found in {bib}")