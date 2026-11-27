"""Pure-Python QR Code SVG generator (Byte mode, Versions 1-10, ECC Level M) for local-only device pairing.

Generates crisp inline SVG QR codes locally without external APIs or third-party dependencies so pairing
codes and tailnet URLs never leave the owner's Mac.
"""

from __future__ import annotations

# GF(256) with primitive polynomial 0x11d (x^8 + x^4 + x^3 + x^2 + 1)
_EXP = [0] * 512
_LOG = [0] * 256
_v = 1
for _i in range(255):
    _EXP[_i] = _v
    _LOG[_v] = _i
    _v <<= 1
    if _v & 0x100:
        _v ^= 0x11D
for _i in range(255, 512):
    _EXP[_i] = _EXP[_i - 255]

# Version table for ECC Level M: (version, total_codewords, ec_per_block, blocks_g1, data_g1, blocks_g2, data_g2, align_centers)
_VERSIONS = [
    (1, 26, 10, 1, 16, 0, 0, ()),
    (2, 44, 16, 1, 28, 0, 0, (6, 18)),
    (3, 70, 26, 1, 44, 0, 0, (6, 22)),
    (4, 100, 18, 2, 32, 0, 0, (6, 26)),
    (5, 134, 24, 2, 43, 0, 0, (6, 30)),
    (6, 172, 16, 4, 27, 0, 0, (6, 34)),
    (7, 196, 18, 4, 31, 0, 0, (6, 22, 38)),
    (8, 242, 22, 2, 38, 2, 39, (6, 24, 42)),
    (9, 292, 22, 3, 36, 2, 37, (6, 26, 46)),
    (10, 346, 26, 4, 43, 1, 44, (6, 28, 50)),
]

# Precomputed format strings for ECC Level M (format indicator 00) and masks 0..7 (15-bit BCH + XOR 0x5412)
_FORMAT_M = [0x5412, 0x5125, 0x5E7C, 0x5B4B, 0x45F9, 0x40CE, 0x4F97, 0x4AA0]
_VERSION_INFO = {7: 0x07C94, 8: 0x085BC, 9: 0x09A99, 10: 0x0A4D3}


def _gf_mul(a: int, b: int) -> int:
    return 0 if a == 0 or b == 0 else _EXP[_LOG[a] + _LOG[b]]


def _rs_gen(degree: int) -> list[int]:
    poly = [1]
    for i in range(degree):
        nxt = [0] * (len(poly) + 1)
        root = _EXP[i]
        for j, coef in enumerate(poly):
            nxt[j] ^= coef
            nxt[j + 1] ^= _gf_mul(coef, root)
        poly = nxt
    return poly


def _rs_encode(data: list[int], ec_len: int) -> list[int]:
    gen = _rs_gen(ec_len)
    rem = list(data) + [0] * ec_len
    for i in range(len(data)):
        coef = rem[i]
        if coef:
            for j in range(1, len(gen)):
                rem[i + j] ^= _gf_mul(gen[j], coef)
    return rem[len(data):]


def _pick_version(n_bytes: int) -> tuple:
    for spec in _VERSIONS:
        ver, _, _, bg1, dg1, bg2, dg2, _ = spec
        cap_bits = (bg1 * dg1 + bg2 * dg2) * 8
        cc_bits = 8 if ver <= 9 else 16
        if 4 + cc_bits + n_bytes * 8 <= cap_bits:
            return spec
    raise ValueError("QR payload too long (max ~210 bytes)")


def _build_codewords(payload: bytes, spec: tuple) -> list[int]:
    ver, _, ec_len, bg1, dg1, bg2, dg2, _ = spec
    total_data = bg1 * dg1 + bg2 * dg2
    bits: list[int] = []

    def push(val: int, width: int) -> None:
        for shift in range(width - 1, -1, -1):
            bits.append((val >> shift) & 1)

    push(0b0100, 4)                          # Byte mode
    push(len(payload), 8 if ver <= 9 else 16)
    for b in payload:
        push(b, 8)
    cap_bits = total_data * 8
    push(0, min(4, cap_bits - len(bits)))    # Terminator
    while len(bits) % 8:
        bits.append(0)
    data = [sum(bits[i + k] << (7 - k) for k in range(8)) for i in range(0, len(bits), 8)]
    pad = (0xEC, 0x11)
    idx = 0
    while len(data) < total_data:
        data.append(pad[idx % 2])
        idx += 1

    blocks: list[tuple[list[int], list[int]]] = []
    pos = 0
    for count, size in ((bg1, dg1), (bg2, dg2)):
        for _ in range(count):
            chunk = data[pos:pos + size]
            pos += size
            blocks.append((chunk, _rs_encode(chunk, ec_len)))

    out: list[int] = []
    max_d = max(len(d) for d, _ in blocks)
    for i in range(max_d):
        for d, _ in blocks:
            if i < len(d):
                out.append(d[i])
    for i in range(ec_len):
        for _, ec in blocks:
            out.append(ec[i])
    return out


def encode_matrix(text: str) -> list[list[bool]]:
    """Encode `text` as a square boolean QR matrix (True = dark module)."""
    payload = text.encode("utf-8")
    spec = _pick_version(len(payload))
    ver, _, _, _, _, _, _, aligns = spec
    size = ver * 4 + 17
    codewords = _build_codewords(payload, spec)

    grid: list[list[int | None]] = [[None] * size for _ in range(size)]
    reserved: list[list[bool]] = [[False] * size for _ in range(size)]

    def set_fn(r: int, c: int, dark: bool) -> None:
        if 0 <= r < size and 0 <= c < size:
            grid[r][c] = 1 if dark else 0
            reserved[r][c] = True

    # Finder patterns + separators
    for fr, fc in ((0, 0), (0, size - 7), (size - 7, 0)):
        for dr in range(-1, 8):
            for dc in range(-1, 8):
                r, c = fr + dr, fc + dc
                if 0 <= r < size and 0 <= c < size:
                    in_outer = 0 <= dr <= 6 and 0 <= dc <= 6
                    dark = in_outer and (
                        dr in (0, 6) or dc in (0, 6) or (2 <= dr <= 4 and 2 <= dc <= 4)
                    )
                    set_fn(r, c, dark)

    # Timing patterns
    for i in range(8, size - 8):
        set_fn(6, i, i % 2 == 0)
        set_fn(i, 6, i % 2 == 0)

    # Alignment patterns
    for ar in aligns:
        for ac in aligns:
            if reserved[ar][ac]:
                continue
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    dark = max(abs(dr), abs(dc)) != 1
                    set_fn(ar + dr, ac + dc, dark)

    # Reserve format info + dark module
    for i in range(9):
        if i != 6:
            set_fn(8, i, False)
            set_fn(i, 8, False)
    for i in range(8):
        set_fn(8, size - 1 - i, False)
        set_fn(size - 1 - i, 8, False)
    set_fn(size - 8, 8, True)

    # Version info (versions >= 7)
    if ver in _VERSION_INFO:
        vbits = _VERSION_INFO[ver]
        for i in range(18):
            bit = bool((vbits >> i) & 1)
            r, c = i // 3, size - 11 + (i % 3)
            set_fn(r, c, bit)
            set_fn(c, r, bit)

    # Place data bits in zigzag columns right-to-left
    bitstream = [(cw >> (7 - b)) & 1 for cw in codewords for b in range(8)]
    bi, col, upward = 0, size - 1, True
    while col > 0:
        if col == 6:
            col -= 1
        rows = range(size - 1, -1, -1) if upward else range(size)
        for r in rows:
            for dc in (0, -1):
                c = col + dc
                if not reserved[r][c]:
                    grid[r][c] = bitstream[bi] if bi < len(bitstream) else 0
                    bi += 1
        col -= 2
        upward = not upward

    # Evaluate masks 0..7 and choose lowest penalty
    masks = (
        lambda r, c: (r + c) % 2 == 0,
        lambda r, c: r % 2 == 0,
        lambda r, c: c % 3 == 0,
        lambda r, c: (r + c) % 3 == 0,
        lambda r, c: (r // 2 + c // 3) % 2 == 0,
        lambda r, c: (r * c) % 2 + (r * c) % 3 == 0,
        lambda r, c: ((r * c) % 2 + (r * c) % 3) % 2 == 0,
        lambda r, c: ((r + c) % 2 + (r * c) % 3) % 2 == 0,
    )

    def apply_mask(m_idx: int) -> list[list[bool]]:
        fn = masks[m_idx]
        mat = [[bool((grid[r][c] or 0) ^ (1 if (not reserved[r][c] and fn(r, c)) else 0))
                for c in range(size)] for r in range(size)]
        fmt = _FORMAT_M[m_idx]
        coords1 = [(8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 7), (8, 8),
                   (7, 8), (5, 8), (4, 8), (3, 8), (2, 8), (1, 8), (0, 8)]
        coords2 = [(size - 1, 8), (size - 2, 8), (size - 3, 8), (size - 4, 8), (size - 5, 8), (size - 6, 8), (size - 7, 8),
                   (8, size - 8), (8, size - 7), (8, size - 6), (8, size - 5), (8, size - 4), (8, size - 3), (8, size - 2), (8, size - 1)]
        for i in range(15):
            bit = bool((fmt >> (14 - i)) & 1)
            r1, c1 = coords1[i]
            r2, c2 = coords2[i]
            mat[r1][c1] = bit
            mat[r2][c2] = bit
        return mat

    def penalty(mat: list[list[bool]]) -> int:
        score = 0
        for line in mat + [[mat[r][c] for r in range(size)] for c in range(size)]:
            run = 1
            for i in range(1, size):
                if line[i] == line[i - 1]:
                    run += 1
                    if run == 5:
                        score += 3
                    elif run > 5:
                        score += 1
                else:
                    run = 1
        for r in range(size - 1):
            for c in range(size - 1):
                if mat[r][c] == mat[r][c + 1] == mat[r + 1][c] == mat[r + 1][c + 1]:
                    score += 3
        dark = sum(cell for row in mat for cell in row)
        score += abs(int(dark * 100 / (size * size)) - 50) // 5 * 10
        return score

    best = min((apply_mask(m) for m in range(8)), key=penalty)
    return best


def to_svg(text: str, *, quiet: int = 3, module_px: int = 4) -> str:
    """Render `text` as a compact, crisp inline SVG QR code with white quiet zone."""
    mat = encode_matrix(text)
    n = len(mat)
    total = (n + quiet * 2) * module_px
    rects: list[str] = []
    for r, row in enumerate(mat):
        c = 0
        while c < n:
            if row[c]:
                start = c
                while c < n and row[c]:
                    c += 1
                x = (start + quiet) * module_px
                y = (r + quiet) * module_px
                w = (c - start) * module_px
                rects.append(f'<rect x="{x}" y="{y}" width="{w}" height="{module_px}"/>')
            else:
                c += 1
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total} {total}" '
        f'width="{total}" height="{total}" role="img" aria-label="Pairing QR code">'
        f'<rect width="100%" height="100%" fill="#ffffff" rx="8"/>'
        f'<g fill="#1c1b18">{"".join(rects)}</g></svg>'
    )
