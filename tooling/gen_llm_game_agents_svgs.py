# -*- coding: utf-8 -*-
"""Generate LLM-Game-Agents post SVGs (UTF-8). Run: python3 tooling/gen_llm_game_agents_svgs.py"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "source" / "images"


def write(name: str, body: str) -> None:
    p = ROOT / name
    p.write_text(body, encoding="utf-8", newline="\n")
    print(p)


SVG_REACT = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 300" width="680" height="300" role="img" aria-labelledby="t">
  <title id="t">ReAct \u5de5\u4f5c\u6d41\u7a0b</title>
  <defs>
    <marker id="ar" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#334155"/>
    </marker>
  </defs>
  <rect fill="#e2e8f0" stroke="#334155" stroke-width="2" x="6" y="6" width="668" height="288" rx="6"/>
  <text x="340" y="32" text-anchor="middle" fill="#0f172a" font-size="15" font-weight="bold" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">ReAct \u5de5\u4f5c\u6d41\u7a0b</text>
  <line x1="6" y1="42" x2="674" y2="42" stroke="#64748b" stroke-width="1.2"/>
  <g font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif" font-size="12" fill="#0f172a">
    <rect fill="#fff" stroke="#334155" x="24" y="58" width="56" height="32" rx="3"/><text x="52" y="78" text-anchor="middle">\u95ee\u9898</text>
    <line x1="80" y1="74" x2="106" y2="74" stroke="#334155" stroke-width="1.5" marker-end="url(#ar)"/>
    <rect fill="#fff" stroke="#334155" x="108" y="58" width="56" height="32" rx="3"/><text x="136" y="78" text-anchor="middle">\u601d\u60f31</text>
    <line x1="164" y1="74" x2="190" y2="74" stroke="#334155" stroke-width="1.5" marker-end="url(#ar)"/>
    <rect fill="#fff" stroke="#334155" x="192" y="58" width="56" height="32" rx="3"/><text x="220" y="78" text-anchor="middle">\u52a8\u4f5c1</text>
    <line x1="248" y1="74" x2="274" y2="74" stroke="#334155" stroke-width="1.5" marker-end="url(#ar)"/>
    <rect fill="#fff" stroke="#334155" x="276" y="58" width="56" height="32" rx="3"/><text x="304" y="78" text-anchor="middle">\u89c2\u5bdf1</text>
    <line x1="304" y1="90" x2="304" y2="118" stroke="#334155" stroke-width="1.5" marker-end="url(#ar)"/>
    <rect fill="#fff" stroke="#334155" x="276" y="122" width="56" height="32" rx="3"/><text x="304" y="142" text-anchor="middle">\u601d\u60f32</text>
    <line x1="332" y1="138" x2="358" y2="138" stroke="#334155" stroke-width="1.5" marker-end="url(#ar)"/>
    <rect fill="#fff" stroke="#334155" x="360" y="122" width="56" height="32" rx="3"/><text x="388" y="142" text-anchor="middle">\u52a8\u4f5c2</text>
    <line x1="416" y1="138" x2="442" y2="138" stroke="#334155" stroke-width="1.5" marker-end="url(#ar)"/>
    <rect fill="#fff" stroke="#334155" x="444" y="122" width="56" height="32" rx="3"/><text x="472" y="142" text-anchor="middle">\u89c2\u5bdf2</text>
    <line x1="472" y1="154" x2="472" y2="182" stroke="#334155" stroke-width="1.5" marker-end="url(#ar)"/>
    <rect fill="#fff" stroke="#334155" x="444" y="186" width="56" height="32" rx="3"/><text x="472" y="206" text-anchor="middle">\u601d\u60f33</text>
    <line x1="500" y1="202" x2="526" y2="202" stroke="#334155" stroke-width="1.5" marker-end="url(#ar)"/>
    <rect fill="#fff" stroke="#334155" x="528" y="186" width="56" height="32" rx="3"/><text x="556" y="206" text-anchor="middle">\u52a8\u4f5c3</text>
    <line x1="584" y1="202" x2="610" y2="202" stroke="#334155" stroke-width="1.5" marker-end="url(#ar)"/>
    <rect fill="#fff" stroke="#334155" x="612" y="186" width="56" height="32" rx="3"/><text x="640" y="206" text-anchor="middle">\u7b54\u6848</text>
  </g>
  <text x="340" y="258" text-anchor="middle" fill="#475569" font-size="11" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u601d\u60f3\uff1a\u63a8\u7406\u89c4\u5212\uff08\u4e0d\u6539\u53d8\u73af\u5883\uff09\u3000\u52a8\u4f5c\uff1a\u4e0e\u73af\u5883\u4ea4\u4e92\u3000\u89c2\u5bdf\uff1a\u73af\u5883\u53cd\u9988</text>
</svg>
"""

SVG_RL = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 200" width="720" height="200" role="img" aria-labelledby="t">
  <title id="t">\u4f20\u7edf RL \u4e0e Reflexion \u5bf9\u6bd4</title>
  <defs>
    <marker id="m" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#334155"/>
    </marker>
  </defs>
  <rect fill="#e2e8f0" stroke="#334155" stroke-width="2" x="6" y="6" width="708" height="188" rx="6"/>
  <text x="360" y="30" text-anchor="middle" fill="#0f172a" font-size="15" font-weight="bold" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u4f20\u7edf RL vs Reflexion</text>
  <line x1="6" y1="40" x2="714" y2="40" stroke="#64748b" stroke-width="1.2"/>
  <g font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif" font-size="11" fill="#0f172a">
    <text x="20" y="62" font-weight="bold" fill="#1e293b">\u4f20\u7edf RL</text>
    <g transform="translate(0, 68)">
      <rect fill="#fff" stroke="#334155" x="16" y="0" width="44" height="26" rx="3"/><text x="38" y="17" text-anchor="middle">\u72b6\u6001</text>
      <line x1="60" y1="13" x2="82" y2="13" stroke="#334155" stroke-width="1.3" marker-end="url(#m)"/>
      <rect fill="#fff" stroke="#334155" x="84" y="0" width="44" height="26" rx="3"/><text x="106" y="17" text-anchor="middle">\u52a8\u4f5c</text>
      <line x1="128" y1="13" x2="150" y2="13" stroke="#334155" stroke-width="1.3" marker-end="url(#m)"/>
      <rect fill="#fff" stroke="#334155" x="152" y="0" width="44" height="26" rx="3"/><text x="174" y="17" text-anchor="middle">\u5956\u52b1</text>
      <line x1="196" y1="13" x2="218" y2="13" stroke="#334155" stroke-width="1.3" marker-end="url(#m)"/>
      <rect fill="#fff" stroke="#334155" x="220" y="0" width="68" height="26" rx="3"/><text x="254" y="17" text-anchor="middle">\u68af\u5ea6\u66f4\u65b0</text>
      <line x1="288" y1="13" x2="310" y2="13" stroke="#334155" stroke-width="1.3" marker-end="url(#m)"/>
      <rect fill="#fff" stroke="#334155" x="312" y="0" width="68" height="26" rx="3"/><text x="346" y="17" text-anchor="middle">\u53c2\u6570\u53d8\u5316</text>
    </g>
    <text x="20" y="128" font-weight="bold" fill="#1e293b">Reflexion</text>
    <g transform="translate(0, 134)">
      <rect fill="#fff" stroke="#334155" x="16" y="0" width="44" height="26" rx="3"/><text x="38" y="17" text-anchor="middle">\u72b6\u6001</text>
      <line x1="60" y1="13" x2="82" y2="13" stroke="#334155" stroke-width="1.3" marker-end="url(#m)"/>
      <rect fill="#fff" stroke="#334155" x="84" y="0" width="44" height="26" rx="3"/><text x="106" y="17" text-anchor="middle">\u52a8\u4f5c</text>
      <line x1="128" y1="13" x2="150" y2="13" stroke="#334155" stroke-width="1.3" marker-end="url(#m)"/>
      <rect fill="#fff" stroke="#334155" x="152" y="0" width="44" height="26" rx="3"/><text x="174" y="17" text-anchor="middle">\u53cd\u9988</text>
      <line x1="196" y1="13" x2="218" y2="13" stroke="#334155" stroke-width="1.3" marker-end="url(#m)"/>
      <rect fill="#fff" stroke="#334155" x="220" y="0" width="68" height="26" rx="3"/><text x="254" y="17" text-anchor="middle">\u8bed\u8a00\u53cd\u601d</text>
      <line x1="288" y1="13" x2="310" y2="13" stroke="#334155" stroke-width="1.3" marker-end="url(#m)"/>
      <rect fill="#fff" stroke="#334155" x="312" y="0" width="68" height="26" rx="3"/><text x="346" y="17" text-anchor="middle">\u8bb0\u5fc6\u5b58\u50a8</text>
      <line x1="346" y1="28" x2="346" y2="48" stroke="#334155" stroke-width="1.3"/>
      <line x1="346" y1="48" x2="430" y2="48" stroke="#334155" stroke-width="1.3" marker-end="url(#m)"/>
      <rect fill="#fef3c7" stroke="#b45309" x="432" y="35" width="72" height="26" rx="3"/><text x="468" y="52" text-anchor="middle">\u4e0b\u6b21\u5c1d\u8bd5</text>
    </g>
  </g>
</svg>
"""

SVG_IO = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 200" width="640" height="200" role="img" aria-labelledby="t">
  <title id="t">Self-Reflection \u8f93\u5165\u4e0e\u8f93\u51fa</title>
  <rect fill="#e2e8f0" stroke="#334155" stroke-width="2" x="6" y="6" width="628" height="188" rx="6"/>
  <text x="320" y="30" text-anchor="middle" fill="#0f172a" font-size="15" font-weight="bold" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u81ea\u6211\u53cd\u601d\uff08Self-Reflection\uff09</text>
  <line x1="6" y1="40" x2="634" y2="40" stroke="#64748b" stroke-width="1.2"/>
  <rect fill="#fff" stroke="#334155" x="24" y="52" width="268" height="130" rx="4"/>
  <text x="158" y="72" text-anchor="middle" fill="#1e293b" font-size="12" font-weight="bold" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u8f93\u5165</text>
  <text x="36" y="92" fill="#0f172a" font-size="11" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\xb7 \u4efb\u52a1\u63cf\u8ff0</text>
  <text x="36" y="110" fill="#0f172a" font-size="11" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\xb7 \u5931\u8d25\u8f68\u8ff9\uff1a\u52a8\u4f5c\u4e0e\u89c2\u5bdf\u5e8f\u5217</text>
  <text x="36" y="128" fill="#0f172a" font-size="11" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\xb7 \u5956\u52b1\u4fe1\u53f7\uff08\u4e8c\u5143\u6216\u6807\u91cf\uff09</text>
  <text x="36" y="146" fill="#0f172a" font-size="11" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\xb7 \u5386\u53f2\u53cd\u601d</text>
  <text x="320" y="114" fill="#64748b" font-size="20" font-family="sans-serif">\u2192</text>
  <rect fill="#fff" stroke="#334155" x="348" y="52" width="268" height="130" rx="4"/>
  <text x="482" y="72" text-anchor="middle" fill="#1e293b" font-size="12" font-weight="bold" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u8f93\u51fa</text>
  <text x="360" y="92" fill="#0f172a" font-size="11" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\xb7 \u9519\u8bef\u8bca\u65ad</text>
  <text x="360" y="110" fill="#0f172a" font-size="11" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\xb7 \u6539\u8fdb\u65b9\u6848</text>
  <text x="360" y="128" fill="#0f172a" font-size="11" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\xb7 \u5177\u4f53\u5efa\u8bae</text>
</svg>
"""

SVG_GEN = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 560 320" width="560" height="320" role="img" aria-labelledby="t">
  <title id="t">Generative Agents \u8bb0\u5fc6\u6d41\u4e0e\u884c\u4e3a\u751f\u6210</title>
  <defs>
    <marker id="ad" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#334155"/>
    </marker>
  </defs>
  <rect fill="#e2e8f0" stroke="#334155" stroke-width="2" x="8" y="8" width="544" height="304" rx="6"/>
  <rect fill="#cbd5e1" stroke="#475569" stroke-width="1.5" x="24" y="28" width="512" height="78" rx="4"/>
  <text x="280" y="52" text-anchor="middle" fill="#0f172a" font-size="13" font-weight="bold" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u8bb0\u5fc6\u6d41\uff08Memory Stream\uff09</text>
  <rect fill="#fff" stroke="#334155" x="36" y="62" width="150" height="36" rx="3"/>
  <text x="111" y="84" text-anchor="middle" fill="#0f172a" font-size="11" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u89c2\u5bdf Observations</text>
  <rect fill="#fff" stroke="#334155" x="205" y="62" width="150" height="36" rx="3"/>
  <text x="280" y="84" text-anchor="middle" fill="#0f172a" font-size="11" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u53cd\u601d Reflections</text>
  <rect fill="#fff" stroke="#334155" x="374" y="62" width="150" height="36" rx="3"/>
  <text x="449" y="84" text-anchor="middle" fill="#0f172a" font-size="11" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u8ba1\u5212 Plans</text>
  <line x1="280" y1="106" x2="280" y2="132" stroke="#334155" stroke-width="2" marker-end="url(#ad)"/>
  <rect fill="#fff" stroke="#334155" x="100" y="136" width="360" height="56" rx="4"/>
  <text x="280" y="158" text-anchor="middle" fill="#1e293b" font-size="12" font-weight="bold" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u8bb0\u5fc6\u68c0\u7d22</text>
  <text x="280" y="178" text-anchor="middle" fill="#475569" font-size="11" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u65f6\u8fd1\u6027 + \u91cd\u8981\u6027 + \u76f8\u5173\u6027</text>
  <line x1="280" y1="192" x2="280" y2="218" stroke="#334155" stroke-width="2" marker-end="url(#ad)"/>
  <rect fill="#fff" stroke="#334155" x="100" y="222" width="360" height="52" rx="4"/>
  <text x="280" y="244" text-anchor="middle" fill="#1e293b" font-size="12" font-weight="bold" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u884c\u4e3a\u751f\u6210</text>
  <text x="280" y="262" text-anchor="middle" fill="#475569" font-size="11" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">Plan, React, Dialogue</text>
</svg>
"""

SVG_TREE = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 200" width="720" height="200" role="img" aria-labelledby="t">
  <title id="t">\u53cd\u601d\u6811\u793a\u610f</title>
  <rect fill="#e2e8f0" stroke="#334155" stroke-width="2" x="6" y="6" width="708" height="188" rx="6"/>
  <text x="360" y="28" text-anchor="middle" fill="#0f172a" font-size="14" font-weight="bold" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u53cd\u601d\u6811\uff08\u53f6=\u89c2\u5bdf\uff0c\u679d=\u53cd\u601d\uff09</text>
  <g stroke="#475569" stroke-width="1.5" fill="none">
    <line x1="360" y1="36" x2="240" y2="78"/><line x1="360" y1="36" x2="480" y2="78"/>
    <line x1="240" y1="106" x2="160" y2="148"/><line x1="240" y1="106" x2="280" y2="148"/>
    <line x1="480" y1="106" x2="400" y2="148"/><line x1="480" y1="106" x2="560" y2="148"/>
  </g>
  <g font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif" font-size="10" fill="#0f172a">
    <rect fill="#f1f5f9" stroke="#334155" x="240" y="40" width="240" height="28" rx="3"/>
    <text x="360" y="58" text-anchor="middle">Klaus \u5bf9\u7814\u7a76\u5145\u6ee1\u70ed\u60c5\uff08\u5143\u53cd\u601d\uff09</text>
    <rect fill="#fff" stroke="#334155" x="120" y="78" width="240" height="28" rx="3"/>
    <text x="240" y="96" text-anchor="middle">Klaus \u81f4\u529b\u4e8e\u7814\u7a76</text>
    <rect fill="#fff" stroke="#334155" x="380" y="78" width="240" height="28" rx="3"/>
    <text x="500" y="96" text-anchor="middle">Klaus \u548c Maria \u6709\u5171\u540c\u5174\u8da3</text>
    <rect fill="#fff" stroke="#334155" x="48" y="148" width="88" height="26" rx="3"/><text x="92" y="165" text-anchor="middle">\u5199\u8bba\u6587</text>
    <rect fill="#fff" stroke="#334155" x="156" y="148" width="88" height="26" rx="3"/><text x="200" y="165" text-anchor="middle">\u8bfb\u4e66</text>
    <rect fill="#fff" stroke="#334155" x="264" y="148" width="88" height="26" rx="3"/><text x="308" y="165" text-anchor="middle">\u8ba8\u8bba\u9879\u76ee</text>
    <rect fill="#fff" stroke="#334155" x="372" y="148" width="112" height="26" rx="3"/><text x="428" y="165" text-anchor="middle">\u56fe\u4e66\u9986\u76f8\u9047</text>
  </g>
</svg>
"""

SVG_IDEAL = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 340" width="640" height="340" role="img" aria-labelledby="t">
  <title id="t">\u7406\u60f3\u667a\u80fd\u4f53\u7ec4\u5408\u67b6\u6784</title>
  <rect fill="#e2e8f0" stroke="#334155" stroke-width="2" x="8" y="8" width="624" height="324" rx="6"/>
  <text x="320" y="34" text-anchor="middle" fill="#0f172a" font-size="15" font-weight="bold" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u7406\u60f3\u667a\u80fd\u4f53\u67b6\u6784\uff08\u7ec4\u5408\uff09</text>
  <line x1="8" y1="44" x2="632" y2="44" stroke="#64748b" stroke-width="1.2"/>
  <rect fill="#fff" stroke="#334155" x="40" y="56" width="560" height="78" rx="4"/>
  <text x="320" y="78" text-anchor="middle" fill="#1e293b" font-size="12" font-weight="bold" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">Generative Agents \u7684\u8bb0\u5fc6\u6d41</text>
  <text x="320" y="98" text-anchor="middle" fill="#475569" font-size="11" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u5b8c\u6574\u7ecf\u5386\u8bb0\u5f55 \xb7 \u591a\u5c42\u6b21\u53cd\u601d \xb7\u793e\u4ea4\u5173\u7cfb\u8ffd\u8e2a</text>
  <text x="320" y="122" text-anchor="middle" fill="#b45309" font-size="18" font-weight="bold" font-family="sans-serif">+</text>
  <rect fill="#fff" stroke="#334155" x="40" y="132" width="560" height="78" rx="4"/>
  <text x="320" y="154" text-anchor="middle" fill="#1e293b" font-size="12" font-weight="bold" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">ReAct \u7684\u63a8\u7406\u2013\u884c\u52a8\u8303\u5f0f</text>
  <text x="320" y="174" text-anchor="middle" fill="#475569" font-size="11" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u601d\u60f3\u4e0e\u52a8\u4f5c\u4ea4\u66ff \xb7 \u4e0e\u5916\u90e8\u73af\u5883\u4ea4\u4e92 \xb7 \u51cf\u5c11\u5e7b\u89c9</text>
  <text x="320" y="198" text-anchor="middle" fill="#b45309" font-size="18" font-weight="bold" font-family="sans-serif">+</text>
  <rect fill="#fff" stroke="#334155" x="40" y="208" width="560" height="78" rx="4"/>
  <text x="320" y="230" text-anchor="middle" fill="#1e293b" font-size="12" font-weight="bold" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">Reflexion \u7684\u5931\u8d25\u53cd\u601d</text>
  <text x="320" y="250" text-anchor="middle" fill="#475569" font-size="11" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u5931\u8d25\u7ecf\u9a8c\u8bed\u8a00\u5316 \xb7 \u9519\u8bef\u8bca\u65ad\u4e0e\u6539\u8fdb\u5efa\u8bae \xb7 \u8de8\u5c1d\u8bd5\u5b66\u4e60</text>
  <text x="320" y="308" text-anchor="middle" fill="#64748b" font-size="10" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">\u7ec4\u5408\uff1a\u8bb0\u5fc6 + ReAct \u884c\u52a8 + Reflexion \u4ece\u9519\u8bef\u4e2d\u5b66\u4e60</text>
</svg>
"""


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    write("llm-game-agents-react-workflow.svg", SVG_REACT)
    write("llm-game-agents-rl-vs-reflexion.svg", SVG_RL)
    write("llm-game-agents-reflexion-reflection-io.svg", SVG_IO)
    write("llm-game-agents-generative-memory-arch.svg", SVG_GEN)
    write("llm-game-agents-reflection-tree.svg", SVG_TREE)
    write("llm-game-agents-ideal-combined-arch.svg", SVG_IDEAL)


if __name__ == "__main__":
    main()
