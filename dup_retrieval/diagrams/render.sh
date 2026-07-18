#!/usr/bin/env bash
# Render all .mmd diagrams to .png via kroki.io (no browser needed, works over SSH).
# Mirrors quality_assesment/diagrams/render.sh for consistency.
D="$(cd "$(dirname "$0")" && pwd)"

for f in "$D"/*.mmd; do
    name="$(basename "$f" .mmd)"
    python3 -c "
import requests, struct, sys
content = open('$f').read()
resp = requests.post('https://kroki.io/mermaid/png', data=content.encode(), headers={'Content-Type': 'text/plain'})
if resp.status_code == 200:
    out = '$D/$name.png'
    open(out, 'wb').write(resp.content)
    data = resp.content
    if data[:4] == b'\x89PNG':
        w, h = struct.unpack('>I', data[16:20])[0], struct.unpack('>I', data[20:24])[0]
        print(f'$name: {w}x{h}')
    else:
        print(f'$name: saved (not PNG?)')
else:
    print(f'$name: ERROR {resp.status_code} - {resp.text[:100]}', file=sys.stderr)
"
done
