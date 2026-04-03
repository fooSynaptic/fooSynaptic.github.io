# -*- coding: utf-8 -*-
"""LLM-Game-Agents 应用扩展篇配图 -> source/images/llm-game-agents-ext-*.svg"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "source" / "images"
F = 'font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif"'
M = """  <defs>
    <marker id="ar" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><polygon points="0 0,8 3,0 6" fill="#334155"/></marker>
  </defs>"""


def svg(vb_w: int, vb_h: int, defs: str, inner: str) -> str:
    return (
        f'<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {vb_w} {vb_h}" width="{vb_w}" height="{vb_h}" role="img">\n{defs}\n{inner}\n</svg>\n'
    )


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)

    # 1 VOYAGER
    voy = svg(
        640,
        340,
        M,
        f"""  <rect fill="#f1f5f9" stroke="#334155" stroke-width="2" x="8" y="8" width="624" height="324" rx="6"/>
  <text x="320" y="32" text-anchor="middle" fill="#0f172a" font-size="14" font-weight="bold" {F}>VOYAGER \u7cfb\u7edf\u67b6\u6784</text>
  <line x1="8" y1="42" x2="632" y2="42" stroke="#64748b"/>
  <rect fill="#e2e8f0" stroke="#475569" x="250" y="56" width="140" height="48" rx="4"/>
  <text x="320" y="76" text-anchor="middle" fill="#0f172a" font-size="11" font-weight="bold" font-family="sans-serif">GPT-4 API</text>
  <text x="320" y="92" text-anchor="middle" fill="#64748b" font-size="9" {F}>(\u9ed1\u76d2\u8c03\u7528)</text>
  <path d="M 520 200 L 520 80 L 395 80" fill="none" stroke="#64748b" stroke-width="1.5" stroke-dasharray="4 3" marker-end="url(#ar)"/>
  <text x="480" y="72" fill="#64748b" font-size="8" {F}>\u53cd\u9988</text>
  <line x1="320" y1="104" x2="142" y2="128" stroke="#334155" stroke-width="1.5" marker-end="url(#ar)"/>
  <line x1="320" y1="104" x2="320" y2="128" stroke="#334155" stroke-width="1.5" marker-end="url(#ar)"/>
  <line x1="320" y1="104" x2="498" y2="128" stroke="#334155" stroke-width="1.5" marker-end="url(#ar)"/>
  <rect fill="#fff" stroke="#334155" x="72" y="132" width="140" height="44" rx="4"/>
  <text x="142" y="150" text-anchor="middle" fill="#0f172a" font-size="10" {F}>\u81ea\u52a8\u8bfe\u7a0b\u751f\u6210</text>
  <text x="142" y="164" text-anchor="middle" fill="#64748b" font-size="8" {F}>(GPT-4 \u63d0\u793a)</text>
  <rect fill="#fff" stroke="#334155" x="250" y="132" width="140" height="44" rx="4"/>
  <text x="320" y="150" text-anchor="middle" fill="#0f172a" font-size="10" {F}>\u4ee3\u7801\u751f\u6210</text>
  <text x="320" y="164" text-anchor="middle" fill="#64748b" font-size="8" {F}>(GPT-4 \u63d0\u793a)</text>
  <rect fill="#fff" stroke="#334155" x="428" y="132" width="140" height="44" rx="4"/>
  <text x="498" y="150" text-anchor="middle" fill="#0f172a" font-size="10" {F}>\u81ea\u6211\u9a8c\u8bc1</text>
  <text x="498" y="164" text-anchor="middle" fill="#64748b" font-size="8" {F}>(GPT-4 \u63d0\u793a)</text>
  <line x1="142" y1="176" x2="142" y2="200" stroke="#334155" marker-end="url(#ar)"/>
  <line x1="320" y1="176" x2="320" y2="200" stroke="#334155" marker-end="url(#ar)"/>
  <line x1="498" y1="176" x2="498" y2="200" stroke="#334155" marker-end="url(#ar)"/>
  <rect fill="#fef3c7" stroke="#b45309" x="72" y="204" width="140" height="40" rx="4"/>
  <text x="142" y="228" text-anchor="middle" fill="#0f172a" font-size="10" {F}>\u4efb\u52a1\u961f\u5217</text>
  <rect fill="#dbeafe" stroke="#1d4ed8" x="250" y="204" width="140" height="40" rx="4"/>
  <text x="320" y="224" text-anchor="middle" fill="#0f172a" font-size="10" {F}>Minecraft</text>
  <text x="320" y="238" text-anchor="middle" fill="#64748b" font-size="9" {F}>\u73af\u5883\u6267\u884c</text>
  <rect fill="#dcfce7" stroke="#15803d" x="428" y="204" width="140" height="40" rx="4"/>
  <text x="498" y="224" text-anchor="middle" fill="#0f172a" font-size="10" {F}>\u6280\u80fd\u5e93</text>
  <text x="498" y="238" text-anchor="middle" fill="#64748b" font-size="9" {F}>(\u5411\u91cf\u6570\u636e\u5e93)</text>""",
    )
    (ROOT / "llm-game-agents-voyager-arch.svg").write_text(voy, encoding="utf-8", newline="\n")

    # 2 PIANO
    piano = svg(
        560,
        300,
        M,
        f"""  <rect fill="#f8fafc" stroke="#334155" stroke-width="2" x="6" y="6" width="548" height="288" rx="6"/>
  <text x="280" y="28" text-anchor="middle" fill="#0f172a" font-size="13" font-weight="bold" {F}>PIANO \u67b6\u6784</text>
  <text x="120" y="52" fill="#64748b" font-size="10" {F}>\u5e76\u53d1\u6a21\u5757</text>
  <rect fill="#fff" stroke="#334155" x="48" y="60" width="100" height="28" rx="3"/><text x="98" y="78" text-anchor="middle" fill="#0f172a" font-size="10" {F}>\u8bb0\u5fc6</text>
  <rect fill="#fff" stroke="#334155" x="48" y="96" width="100" height="28" rx="3"/><text x="98" y="114" text-anchor="middle" fill="#0f172a" font-size="10" {F}>\u793e\u4f1a</text>
  <rect fill="#fff" stroke="#334155" x="48" y="132" width="100" height="28" rx="3"/><text x="98" y="150" text-anchor="middle" fill="#0f172a" font-size="10" {F}>\u76ee\u6807</text>
  <rect fill="#fff" stroke="#334155" x="48" y="168" width="100" height="28" rx="3"/><text x="98" y="186" text-anchor="middle" fill="#0f172a" font-size="10" {F}>\u52a8\u4f5c</text>
  <line x1="148" y1="74" x2="220" y2="120" stroke="#334155" marker-end="url(#ar)"/>
  <line x1="148" y1="110" x2="220" y2="130" stroke="#334155" marker-end="url(#ar)"/>
  <line x1="148" y1="146" x2="220" y2="140" stroke="#334155" marker-end="url(#ar)"/>
  <line x1="148" y1="182" x2="220" y2="150" stroke="#334155" marker-end="url(#ar)"/>
  <rect fill="#e2e8f0" stroke="#475569" x="220" y="88" width="120" height="100" rx="4"/>
  <text x="280" y="108" text-anchor="middle" fill="#0f172a" font-size="10" font-weight="bold" {F}>\u8ba4\u77e5\u63a7\u5236\u5668</text>
  <text x="280" y="124" text-anchor="middle" fill="#64748b" font-size="9" {F}>\u4fe1\u606f\u7efc\u5408</text>
  <text x="280" y="142" text-anchor="middle" fill="#64748b" font-size="9" {F}>\u2193 \u9ad8\u5c42\u51b3\u7b56</text>
  <text x="280" y="160" text-anchor="middle" fill="#64748b" font-size="9" {F}>\u2193 \u51b3\u7b56\u5e7f\u64ad</text>
  <line x1="280" y1="188" x2="280" y2="212" stroke="#334155" marker-end="url(#ar)"/>
  <rect fill="#fff" stroke="#334155" x="220" y="216" width="120" height="44" rx="4"/>
  <text x="280" y="236" text-anchor="middle" fill="#0f172a" font-size="10" {F}>\u8f93\u51fa\u6a21\u5757</text>
  <text x="280" y="250" text-anchor="middle" fill="#64748b" font-size="8" {F}>\u8bf4\u8bdd / \u52a8\u4f5c / \u2026</text>
  <path d="M 340 238 Q 420 260 420 200 Q 420 120 148 120" fill="none" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#ar)"/>
  <text x="400" y="100" fill="#94a3b8" font-size="8" {F}>\u53cd\u9988</text>
  <text x="400" y="52" fill="#64748b" font-size="9" {F}>(\u74f6\u9888)</text>""",
    )
    (ROOT / "llm-game-agents-piano-arch.svg").write_text(piano, encoding="utf-8", newline="\n")

    # 3 \u6cbb\u7597\u95ed\u73af
    care = svg(
        520,
        240,
        M,
        f"""  <rect fill="#f8fafc" stroke="#334155" stroke-width="2" x="8" y="8" width="504" height="224" rx="6"/>
  <text x="260" y="32" text-anchor="middle" fill="#0f172a" font-size="13" font-weight="bold" {F}>\u6cbb\u7597\u95ed\u73af</text>
  <text x="260" y="56" text-anchor="middle" fill="#475569" font-size="10" {F}>1\u75be\u75c5\u53d1\u4f5c \u2192 2\u5206\u8bca \u2192 3\u6302\u53f7</text>
  <line x1="260" y1="64" x2="260" y2="88" stroke="#334155" marker-end="url(#ar)"/>
  <text x="260" y="108" text-anchor="middle" fill="#0f172a" font-size="10" {F}>4 \u5c31\u8bca \u2192 5 \u68c0\u67e5 \u2192 6 \u8bca\u65ad</text>
  <line x1="260" y1="116" x2="260" y2="132" stroke="#334155" marker-end="url(#ar)"/>
  <text x="260" y="152" text-anchor="middle" fill="#0f172a" font-size="10" {F}>7 \u53d6\u836f \u2192 8 \u5eb7\u590d\u53cd\u9988</text>
  <path d="M 120 168 Q 80 200 260 200 Q 440 200 400 168" fill="none" stroke="#64748b" stroke-width="1.5" marker-end="url(#ar)"/>
  <text x="260" y="198" text-anchor="middle" fill="#64748b" font-size="9" {F}>\u95ed\u73af</text>
  <text x="260" y="220" text-anchor="middle" fill="#94a3b8" font-size="8" {F}>\u989d\u5916\uff1a\u533b\u751f\u667a\u80fd\u4f53\u95f2\u65f6\u9605\u8bfb\u533b\u5b66\u4e66\u7c4d</text>""",
    )
    (ROOT / "llm-game-agents-care-loop.svg").write_text(care, encoding="utf-8", newline="\n")

    # 4 MedAgent-Zero
    ma = svg(
        440,
        280,
        M,
        f"""  <rect fill="#f8fafc" stroke="#334155" stroke-width="2" x="6" y="6" width="428" height="268" rx="6"/>
  <text x="220" y="30" text-anchor="middle" fill="#0f172a" font-size="13" font-weight="bold" {F}>MedAgent-Zero \u8fdb\u5316\u6d41\u7a0b</text>
  <text x="220" y="52" text-anchor="middle" fill="#475569" font-size="10" {F}>1. \u6cbb\u7597\u60a3\u8005\u667a\u80fd\u4f53</text>
  <line x1="220" y1="56" x2="220" y2="72" stroke="#334155" marker-end="url(#ar)"/>
  <text x="220" y="88" text-anchor="middle" fill="#475569" font-size="10" {F}>2. \u60a3\u8005\u53cd\u9988\uff08\u5eb7\u590d / \u672a\u5eb7\u590d\uff09</text>
  <line x1="220" y1="92" x2="220" y2="108" stroke="#334155" marker-end="url(#ar)"/>
  <rect fill="#dcfce7" stroke="#15803d" x="48" y="116" width="160" height="56" rx="4"/>
  <text x="128" y="136" text-anchor="middle" fill="#0f172a" font-size="10" {F}>\u6210\u529f\u6848\u4f8b</text>
  <text x="128" y="154" text-anchor="middle" fill="#64748b" font-size="9" {F}>\u5b58\u50a8\u53c2\u8003 \xb7 \u68c0\u7d22</text>
  <rect fill="#fee2e2" stroke="#b91c1c" x="232" y="116" width="160" height="56" rx="4"/>
  <text x="312" y="136" text-anchor="middle" fill="#0f172a" font-size="10" {F}>\u5931\u8d25\u6848\u4f8b</text>
  <text x="312" y="154" text-anchor="middle" fill="#64748b" font-size="9" {F}>\u53cd\u601d\u7ecf\u9a8c \xb7 \u907f\u514d\u91cd\u590d</text>
  <line x1="220" y1="172" x2="220" y2="196" stroke="#334155" marker-end="url(#ar)"/>
  <text x="220" y="212" text-anchor="middle" fill="#475569" font-size="10" {F}>3. \u9605\u8bfb\u533b\u5b66\u6559\u6750\u5de9\u56fa</text>
  <line x1="220" y1="216" x2="220" y2="228" stroke="#334155" marker-end="url(#ar)"/>
  <text x="220" y="248" text-anchor="middle" fill="#0f172a" font-size="10" font-weight="bold" {F}>4. \u80fd\u529b\u6301\u7eed\u63d0\u5347</text>""",
    )
    (ROOT / "llm-game-agents-medagent-zero.svg").write_text(ma, encoding="utf-8", newline="\n")

    # 5 SEAL \u516c\u5f0f
    seal = svg(
        560,
        72,
        "",
        f"""  <rect fill="#f1f5f9" stroke="#cbd5e1" x="4" y="8" width="552" height="52" rx="4"/>
  <text x="280" y="40" text-anchor="middle" fill="#0f172a" font-size="11" {F}>\u9886\u57df\u5de5\u4f5c\u6d41\u7a0b \u2192 \u4eff\u771f\u7cfb\u7edf \u2192 \u81ea\u52a8\u751f\u6210\u6570\u636e \u2192 \u667a\u80fd\u4f53\u8fdb\u5316</text>""",
    )
    (ROOT / "llm-game-agents-seal-pipeline.svg").write_text(seal, encoding="utf-8", newline="\n")

    # 6 \u6280\u672f\u6f14\u8fdb\u6811
    tree = svg(
        560,
        220,
        M,
        f"""  <text x="280" y="22" text-anchor="middle" fill="#0f172a" font-size="12" font-weight="bold" {F}>\u6280\u672f\u6f14\u8fdb\u8def\u7ebf</text>
  <text x="80" y="48" fill="#475569" font-size="10" font-weight="bold" {F}>\u57fa\u7840\u6846\u67b6 (2022-23)</text>
  <text x="40" y="68" fill="#0f172a" font-size="9" {F}>\u251c ReAct</text>
  <text x="40" y="84" fill="#0f172a" font-size="9" {F}>\u251c Reflexion</text>
  <text x="40" y="100" fill="#0f172a" font-size="9" {F}>\u2514 Generative Agents</text>
  <text x="280" y="48" fill="#475569" font-size="10" font-weight="bold" {F}>\u5e94\u7528\u6269\u5c55 (2023-24)</text>
  <text x="220" y="68" fill="#0f172a" font-size="9" {F}>\u251c VOYAGER</text>
  <text x="220" y="84" fill="#0f172a" font-size="9" {F}>\u251c Project Sid</text>
  <text x="220" y="100" fill="#0f172a" font-size="9" {F}>\u2514 Agent Hospital</text>
  <text x="480" y="48" fill="#475569" font-size="10" font-weight="bold" {F}>\u672a\u6765 (2025+)</text>
  <text x="400" y="68" fill="#0f172a" font-size="9" {F}>\u251c Agent OS / LangGraph</text>
  <text x="400" y="84" fill="#0f172a" font-size="9" {F}>\u251c \u591a\u6a21\u6001\u878d\u5408</text>
  <text x="400" y="100" fill="#0f172a" font-size="9" {F}>\u2514 \u5546\u4e1a\u5316\u90e8\u7f72</text>
  <line x1="120" y1="88" x2="200" y2="88" stroke="#cbd5e1"/>
  <line x1="340" y1="88" x2="380" y2="88" stroke="#cbd5e1"/>""",
    )
    (ROOT / "llm-game-agents-tech-roadmap.svg").write_text(tree, encoding="utf-8", newline="\n")

    # 7 \u7406\u60f3\u7ec4\u5408
    ideal = svg(
        520,
        200,
        M,
        f"""  <text x="260" y="24" text-anchor="middle" fill="#0f172a" font-size="12" font-weight="bold" {F}>\u7406\u60f3\u667a\u80fd\u4f53\u7ec4\u5408\uff08\u6587\u5b57\u793a\u610f\uff09</text>
  <text x="260" y="52" text-anchor="middle" fill="#475569" font-size="10" {F}>= VOYAGER \u6280\u80fd\u5e93</text>
  <text x="260" y="72" text-anchor="middle" fill="#475569" font-size="10" {F}>+ Project Sid \u793e\u4f1a\u610f\u8bc6</text>
  <text x="260" y="92" text-anchor="middle" fill="#475569" font-size="10" {F}>+ Agent Hospital \u8fdb\u5316\u673a\u5236</text>
  <text x="260" y="112" text-anchor="middle" fill="#475569" font-size="10" {F}>+ Generative Agents \u8bb0\u5fc6\u7cfb\u7edf</text>""",
    )
    (ROOT / "llm-game-agents-ideal-stack.svg").write_text(ideal, encoding="utf-8", newline="\n")
    print("wrote 7 SVGs under", ROOT)


if __name__ == "__main__":
    main()
