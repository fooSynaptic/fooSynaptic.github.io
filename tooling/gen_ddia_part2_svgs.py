# -*- coding: utf-8 -*-
"""Generate DDIA Part2 diagrams under source/images/ddia/part2/. Run from repo root:
   python3 tooling/gen_ddia_part2_svgs.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "source" / "images" / "ddia" / "part2"


def write(name: str, body: str) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / name).write_text(body, encoding="utf-8", newline="\n")
    print(ROOT / name)


# --- 1. \u4e3b\u4ece\u590d\u5236 ---
SVG_LEADER_FOLLOWER = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 520 320" width="520" height="320" role="img">
  <title>\u4e3b\u4ece\u590d\u5236\u67b6\u6784</title>
  <defs>
    <marker id="ah" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><polygon points="0 0,8 3,0 6" fill="#334155"/></marker>
  </defs>
  <rect fill="#e2e8f0" stroke="#334155" stroke-width="2" x="6" y="6" width="508" height="308" rx="6"/>
  <text x="260" y="28" text-anchor="middle" fill="#0f172a" font-size="14" font-weight="bold" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u4e3b\u4ece\u590d\u5236\u67b6\u6784</text>
  <line x1="6" y1="36" x2="514" y2="36" stroke="#64748b"/>
  <text x="260" y="54" text-anchor="middle" fill="#0f172a" font-size="11" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u5ba2\u6237\u7aef\u5199\u5165</text>
  <line x1="260" y="58" x2="260" y2="78" stroke="#334155" stroke-width="1.5" marker-end="url(#ah)"/>
  <rect fill="#fff" stroke="#334155" x="200" y="82" width="120" height="44" rx="4"/>
  <text x="260" y="102" text-anchor="middle" fill="#0f172a" font-size="11" font-weight="bold" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u4e3b\u8282\u70b9</text>
  <text x="260" y="118" text-anchor="middle" fill="#475569" font-size="10" font-family="sans-serif">(Leader)</text>
  <line x1="320" y1="104" x2="380" y2="104" stroke="#334155" marker-end="url(#ah)"/>
  <text x="350" y="98" text-anchor="middle" fill="#64748b" font-size="9" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u590d\u5236\u65e5\u5fd7</text>
  <rect fill="#fff" stroke="#334155" x="384" y="82" width="110" height="44" rx="4"/>
  <text x="439" y="102" text-anchor="middle" fill="#0f172a" font-size="11" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u4ece\u8282\u70b91</text>
  <text x="439" y="118" text-anchor="middle" fill="#475569" font-size="9" font-family="sans-serif">(Follower)</text>
  <line x1="260" y1="126" x2="260" y2="158" stroke="#334155" stroke-width="1.5" marker-end="url(#ah)"/>
  <text x="228" y="146" fill="#64748b" font-size="9" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u590d\u5236\u65e5\u5fd7</text>
  <rect fill="#fff" stroke="#334155" x="200" y="162" width="120" height="40" rx="4"/>
  <text x="260" y="186" text-anchor="middle" fill="#0f172a" font-size="11" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u4ece\u8282\u70b92</text>
  <line x1="439" y1="126" x2="439" y2="162" stroke="#334155" stroke-width="1.5" marker-end="url(#ah)"/>
  <text x="455" y="148" fill="#64748b" font-size="9" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u590d\u5236\u65e5\u5fd7</text>
  <rect fill="#fff" stroke="#334155" x="384" y="162" width="110" height="40" rx="4"/>
  <text x="439" y="186" text-anchor="middle" fill="#0f172a" font-size="11" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u4ece\u8282\u70b93</text>
  <text x="260" y="232" text-anchor="middle" fill="#475569" font-size="11" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u5ba2\u6237\u7aef\u53ef\u4ece\u4efb\u610f\u8282\u70b9\u8bfb\u53d6</text>
</svg>
"""

# --- 2. \u591a\u4e3b\u6570\u636e\u4e2d\u5fc3 ---
SVG_MULTI_DC = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 420 120" width="420" height="120" role="img">
  <title>\u591a\u4e3b\u590d\u5236\uff1a\u6570\u636e\u4e2d\u5fc3</title>
  <defs>
    <marker id="m1" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><polygon points="0 0,8 3,0 6" fill="#334155"/></marker>
  </defs>
  <rect fill="#fff" stroke="#334155" stroke-width="2" x="24" y="28" width="140" height="64" rx="6"/>
  <text x="94" y="52" text-anchor="middle" fill="#0f172a" font-size="12" font-weight="bold" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u6570\u636e\u4e2d\u5fc31</text>
  <text x="94" y="74" text-anchor="middle" fill="#475569" font-size="10" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">(\u4e3b\u8282\u70b91)</text>
  <rect fill="#fff" stroke="#334155" stroke-width="2" x="256" y="28" width="140" height="64" rx="6"/>
  <text x="326" y="52" text-anchor="middle" fill="#0f172a" font-size="12" font-weight="bold" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u6570\u636e\u4e2d\u5fc32</text>
  <text x="326" y="74" text-anchor="middle" fill="#475569" font-size="10" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">(\u4e3b\u8282\u70b92)</text>
  <line x1="168" y1="58" x2="244" y2="58" stroke="#334155" stroke-width="1.5" marker-end="url(#m1)"/>
  <line x1="244" y1="62" x2="168" y2="62" stroke="#334155" stroke-width="1.5" marker-end="url(#m1)"/>
  <text x="206" y="54" text-anchor="middle" fill="#b45309" font-size="10" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u5f02\u6b65</text>
</svg>
"""

# --- 3. 2PC ---
SVG_2PC = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 460 200" width="460" height="200" role="img">
  <title>\u4e24\u9636\u6bb5\u63d0\u4ea4\uff082PC\uff09</title>
  <defs>
    <marker id="a2" markerWidth="7" markerHeight="7" refX="6" refY="2.5" orient="auto"><polygon points="0 0,7 2.5,0 5" fill="#334155"/></marker>
  </defs>
  <rect fill="#f8fafc" stroke="#334155" stroke-width="1.5" x="6" y="6" width="448" height="188" rx="6"/>
  <text x="20" y="28" fill="#1e293b" font-size="12" font-weight="bold" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u9636\u6bb51\uff08\u51c6\u5907\uff09</text>
  <rect fill="#e2e8f0" stroke="#475569" x="24" y="40" width="72" height="28" rx="3"/><text x="60" y="58" text-anchor="middle" font-size="10" fill="#0f172a" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u534f\u8c03\u8005</text>
  <line x1="96" y1="54" x2="268" y2="54" stroke="#334155" marker-end="url(#a2)"/>
  <text x="175" y="48" text-anchor="middle" fill="#64748b" font-size="9" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u51c6\u5907</text>
  <rect fill="#fff" stroke="#334155" x="276" y="40" width="88" height="28" rx="3"/><text x="320" y="58" text-anchor="middle" font-size="10" fill="#0f172a" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u53c2\u4e0e\u8005\u4eec</text>
  <line x1="268" y1="76" x2="100" y2="76" stroke="#94a3b8" marker-end="url(#a2)"/>
  <text x="180" y="88" text-anchor="middle" fill="#64748b" font-size="9" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u51c6\u5907\u597d</text>
  <text x="20" y="112" fill="#1e293b" font-size="12" font-weight="bold" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u9636\u6bb52\uff08\u63d0\u4ea4\uff09</text>
  <rect fill="#e2e8f0" stroke="#475569" x="24" y="122" width="72" height="28" rx="3"/><text x="60" y="140" text-anchor="middle" font-size="10" fill="#0f172a" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u534f\u8c03\u8005</text>
  <line x1="96" y1="136" x2="268" y2="136" stroke="#334155" marker-end="url(#a2)"/>
  <text x="175" y="130" text-anchor="middle" fill="#64748b" font-size="9" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u63d0\u4ea4</text>
  <rect fill="#fff" stroke="#334155" x="276" y="122" width="88" height="28" rx="3"/><text x="320" y="140" text-anchor="middle" font-size="10" fill="#0f172a" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u53c2\u4e0e\u8005\u4eec</text>
  <line x1="268" y1="158" x2="100" y2="158" stroke="#94a3b8" marker-end="url(#a2)"/>
  <text x="180" y="170" text-anchor="middle" fill="#64748b" font-size="9" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u5b8c\u6210</text>
</svg>
"""

# --- 4. \u4e00\u81f4\u6027\u8c31 ---
SVG_SPECTRUM = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 560 100" width="560" height="100" role="img">
  <title>\u4e00\u81f4\u6027\u6a21\u578b\u5f3a\u5ea6\u8c31</title>
  <text x="40" y="28" fill="#0f172a" font-size="12" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u5f31</text>
  <text x="500" y="28" text-anchor="end" fill="#0f172a" font-size="12" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u5f3a</text>
  <line x1="60" y1="36" x2="500" y2="36" stroke="#334155" stroke-width="2"/>
  <line x1="100" y1="32" x2="100" y2="44" stroke="#64748b"/>
  <line x1="220" y1="32" x2="220" y2="44" stroke="#64748b"/>
  <line x1="340" y1="32" x2="340" y2="44" stroke="#64748b"/>
  <line x1="460" y1="32" x2="460" y2="44" stroke="#64748b"/>
  <text x="100" y="64" text-anchor="middle" fill="#475569" font-size="10" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u6700\u7ec8\u4e00\u81f4\u6027</text>
  <text x="220" y="64" text-anchor="middle" fill="#475569" font-size="10" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u56e0\u679c\u4e00\u81f4\u6027</text>
  <text x="340" y="64" text-anchor="middle" fill="#475569" font-size="10" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u987a\u5e8f\u4e00\u81f4\u6027</text>
  <text x="460" y="64" text-anchor="middle" fill="#475569" font-size="10" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u7ebf\u6027\u4e00\u81f4\u6027</text>
</svg>
"""

# --- 5. \u7ebf\u6027\u4e00\u81f4\u6027\u65f6\u95f4\u7ebf ---
SVG_LINEAR_TIME = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 520 150" width="520" height="150" role="img">
  <title>\u7ebf\u6027\u4e00\u81f4\u6027\u65f6\u95f4\u7ebf</title>
  <text x="16" y="22" fill="#1e293b" font-size="11" font-weight="bold" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u65f6\u95f4\u7ebf:</text>
  <line x1="80" y1="34" x2="480" y2="34" stroke="#334155" stroke-width="1.5"/>
  <polygon points="480,34 472,30 472,38" fill="#334155"/>
  <text x="16" y="58" fill="#0f172a" font-size="11" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u5ba2\u6237\u7aefA:</text>
  <rect fill="#93c5fd" stroke="#1d4ed8" x="100" y="48" width="120" height="18" rx="2"/>
  <text x="160" y="60" text-anchor="middle" fill="#0f172a" font-size="9" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u5199\u5165 x=1</text>
  <text x="16" y="88" fill="#0f172a" font-size="11" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u5ba2\u6237\u7aefB:</text>
  <rect fill="#86efac" stroke="#15803d" x="140" y="78" width="100" height="18" rx="2"/>
  <text x="190" y="90" text-anchor="middle" fill="#0f172a" font-size="9" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u8bfb\u53d6 x</text>
  <text x="250" y="90" fill="#15803d" font-size="9" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u2192 \u5fc5\u987b\u8fd4\u56de 1</text>
  <text x="16" y="118" fill="#0f172a" font-size="11" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u5ba2\u6237\u7aefC:</text>
  <rect fill="#86efac" stroke="#15803d" x="180" y="108" width="100" height="18" rx="2"/>
  <text x="230" y="120" text-anchor="middle" fill="#0f172a" font-size="9" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u8bfb\u53d6 x</text>
  <text x="290" y="120" fill="#15803d" font-size="9" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u2192 \u5fc5\u987b\u8fd4\u56de 1</text>
  <text x="260" y="142" text-anchor="middle" fill="#64748b" font-size="10" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u4e00\u65e6\u4efb\u4f55\u5ba2\u6237\u7aef\u8bfb\u5230\u65b0\u503c\uff0c\u540e\u7eed\u8bfb\u53d6\u5747\u987b\u8fd4\u56de\u65b0\u503c</text>
</svg>
"""

# --- 6. CAP ---
SVG_CAP = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 360 220" width="360" height="220" role="img">
  <title>CAP \u5b9a\u7406</title>
  <polygon fill="#f1f5f9" stroke="#334155" stroke-width="2" points="180,24 48,188 312,188"/>
  <text x="180" y="56" text-anchor="middle" fill="#0f172a" font-size="13" font-weight="bold" font-family="sans-serif">C (Consistency)</text>
  <text x="68" y="182" fill="#0f172a" font-size="11" font-weight="bold" font-family="sans-serif">A</text>
  <text x="52" y="196" fill="#475569" font-size="9" font-family="sans-serif">(Availability)</text>
  <text x="272" y="182" text-anchor="end" fill="#0f172a" font-size="11" font-weight="bold" font-family="sans-serif">P</text>
  <text x="298" y="196" text-anchor="end" fill="#475569" font-size="9" font-family="sans-serif">(Partition Tolerance)</text>
  <text x="180" y="118" text-anchor="middle" fill="#64748b" font-size="11" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u53ea\u80fd\u9009 2 \u4e2a</text>
  <text x="180" y="188" text-anchor="middle" fill="#475569" font-size="9" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">CP\uff1a\u4fdd\u8bc1\u4e00\u81f4\u6027\uff0c\u727a\u7272\u53ef\u7528\u6027</text>
  <text x="180" y="202" text-anchor="middle" fill="#475569" font-size="9" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">AP\uff1a\u4fdd\u8bc1\u53ef\u7528\u6027\uff0c\u727a\u7272\u4e00\u81f4\u6027</text>
</svg>
"""

# --- 7. Raft ---
SVG_RAFT = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 340 200" width="340" height="200" role="img">
  <title>Raft \u9886\u5bfc\u8005\u4e0e\u8ddf\u968f\u8005</title>
  <defs>
    <marker id="ar7" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><polygon points="0 0,8 3,0 6" fill="#334155"/></marker>
  </defs>
  <rect fill="#fff" stroke="#334155" stroke-width="2" x="110" y="16" width="120" height="52" rx="4"/>
  <text x="170" y="36" text-anchor="middle" fill="#0f172a" font-size="12" font-weight="bold" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u9886\u5bfc\u8005</text>
  <text x="170" y="54" text-anchor="middle" fill="#475569" font-size="10" font-family="sans-serif">(Leader)</text>
  <text x="248" y="42" fill="#64748b" font-size="10" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u2190 \u5ba2\u6237\u7aef\u8bf7\u6c42</text>
  <line x1="170" y1="68" x2="170" y2="96" stroke="#334155" stroke-width="1.5" marker-end="url(#ar7)"/>
  <text x="188" y="88" fill="#64748b" font-size="9" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u590d\u5236\u65e5\u5fd7</text>
  <rect fill="#fff" stroke="#334155" x="48" y="104" width="110" height="48" rx="4"/>
  <text x="103" y="124" text-anchor="middle" fill="#0f172a" font-size="11" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u8ddf\u968f\u80051</text>
  <text x="103" y="142" text-anchor="middle" fill="#475569" font-size="9" font-family="sans-serif">(Follower)</text>
  <rect fill="#fff" stroke="#334155" x="182" y="104" width="110" height="48" rx="4"/>
  <text x="237" y="124" text-anchor="middle" fill="#0f172a" font-size="11" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u8ddf\u968f\u80052</text>
  <text x="237" y="142" text-anchor="middle" fill="#475569" font-size="9" font-family="sans-serif">(Follower)</text>
  <line x1="130" y1="104" x2="160" y2="72" stroke="#cbd5e1" stroke-width="1"/>
  <line x1="210" y1="104" x2="180" y2="72" stroke="#cbd5e1" stroke-width="1"/>
</svg>
"""

# --- 8. Paxos ---
SVG_PAXOS = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 480 200" width="480" height="200" role="img">
  <title>Paxos \u4e24\u9636\u6bb5</title>
  <defs>
    <marker id="ap" markerWidth="7" markerHeight="7" refX="6" refY="2.5" orient="auto"><polygon points="0 0,7 2.5,0 5" fill="#334155"/></marker>
  </defs>
  <rect fill="#f8fafc" stroke="#334155" stroke-width="1.5" x="6" y="6" width="468" height="188" rx="6"/>
  <text x="20" y="28" fill="#1e293b" font-size="12" font-weight="bold" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u9636\u6bb51\uff08Prepare\uff09</text>
  <rect fill="#e2e8f0" stroke="#475569" x="24" y="40" width="72" height="28" rx="3"/><text x="60" y="58" text-anchor="middle" font-size="10" fill="#0f172a" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u63d0\u8bae\u8005</text>
  <line x1="96" y1="54" x2="288" y2="54" stroke="#334155" marker-end="url(#ap)"/>
  <text x="188" y="48" text-anchor="middle" fill="#64748b" font-size="9" font-family="monospace">Prepare(n)</text>
  <rect fill="#fff" stroke="#334155" x="296" y="40" width="72" height="28" rx="3"/><text x="332" y="58" text-anchor="middle" font-size="10" fill="#0f172a" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u63a5\u53d7\u8005</text>
  <line x1="288" y1="76" x2="100" y2="76" stroke="#94a3b8" marker-end="url(#ap)"/>
  <text x="190" y="88" text-anchor="middle" fill="#64748b" font-size="9" font-family="sans-serif">Promise</text>
  <text x="20" y="112" fill="#1e293b" font-size="12" font-weight="bold" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u9636\u6bb52\uff08Accept\uff09</text>
  <rect fill="#e2e8f0" stroke="#475569" x="24" y="122" width="72" height="28" rx="3"/><text x="60" y="140" text-anchor="middle" font-size="10" fill="#0f172a" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u63d0\u8bae\u8005</text>
  <line x1="96" y1="136" x2="288" y2="136" stroke="#334155" marker-end="url(#ap)"/>
  <text x="188" y="130" text-anchor="middle" fill="#64748b" font-size="9" font-family="monospace">Accept(n,v)</text>
  <rect fill="#fff" stroke="#334155" x="296" y="122" width="72" height="28" rx="3"/><text x="332" y="140" text-anchor="middle" font-size="10" fill="#0f172a" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u63a5\u53d7\u8005</text>
  <line x1="288" y1="158" x2="100" y2="158" stroke="#94a3b8" marker-end="url(#ap)"/>
  <text x="190" y="170" text-anchor="middle" fill="#64748b" font-size="9" font-family="sans-serif">Accepted</text>
</svg>
"""


def main():
    write("leader-follower-replication.svg", SVG_LEADER_FOLLOWER)
    write("multi-datacenter-multi-master.svg", SVG_MULTI_DC)
    write("two-phase-commit.svg", SVG_2PC)
    write("consistency-spectrum.svg", SVG_SPECTRUM)
    write("linearizability-timeline.svg", SVG_LINEAR_TIME)
    write("cap-theorem.svg", SVG_CAP)
    write("raft-leader-followers.svg", SVG_RAFT)
    write("paxos-two-phases.svg", SVG_PAXOS)


if __name__ == "__main__":
    main()
