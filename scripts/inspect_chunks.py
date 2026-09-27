import json
from pathlib import Path

chunks = json.loads(Path("data/index/chunks.json").read_text(encoding="utf-8"))
print(f"Total chunks: {len(chunks)}")

sections = {}
for c in chunks:
    s = c["section"]
    sections[s] = sections.get(s, 0) + 1

print(f"Sections ({len(sections)}):")
for s, count in sorted(sections.items()):
    print(f"  [{count}] {s}")

print("\nSample chunk 0:")
c0 = chunks[0]
print(f"  document : {c0['document']}")
print(f"  section  : {c0['section']}")
print(f"  chunk_id : {c0['chunk_id']}")
print(f"  text[:200]: {c0['text'][:200]}")
