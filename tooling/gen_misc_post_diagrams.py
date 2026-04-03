# -*- coding: utf-8 -*-
"""Misc post diagrams -> source/images/. Run: python3 tooling/gen_misc_post_diagrams.py"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "source" / "images"
F = 'font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif"'
MARK = """  <defs>
    <marker id="ar" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><polygon points="0 0,8 3,0 6" fill="#334155"/></marker>
  </defs>"""


def svg(w: int, h: int, defs: str, inner: str) -> str:
    return (
        f'<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img">\n{defs}\n{inner}\n</svg>\n'
    )


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)

    # --- LLM \u6e38\u620f\u667a\u80fd\u4f53\u6280\u672f\u6808 ---
    stack = svg(
        520,
        320,
        MARK,
        f"""  <rect fill="#f1f5f9" stroke="#334155" stroke-width="2" x="8" y="8" width="504" height="304" rx="6"/>
  <text x="260" y="30" text-anchor="middle" fill="#0f172a" font-size="14" font-weight="bold" {F}>LLM \u6e38\u620f\u667a\u80fd\u4f53\u6280\u672f\u6808</text>
  <line x1="8" y1="40" x2="512" y2="40" stroke="#64748b"/>
  <rect fill="#fff" stroke="#334155" x="120" y="56" width="280" height="44" rx="4"/>
  <text x="260" y="74" text-anchor="middle" fill="#0f172a" font-size="11" font-weight="bold" {F}>\u5e94\u7528\u5c42</text>
  <text x="260" y="90" text-anchor="middle" fill="#64748b" font-size="10" {F}>\u6e38\u620f / \u6a21\u62df / \u673a\u5668\u4eba</text>
  <line x1="260" y1="100" x2="260" y2="116" stroke="#334155" marker-end="url(#ar)"/>
  <rect fill="#e2e8f0" stroke="#475569" x="120" y="120" width="280" height="44" rx="4"/>
  <text x="260" y="138" text-anchor="middle" fill="#0f172a" font-size="11" font-weight="bold" {F}>\u667a\u80fd\u4f53\u6846\u67b6</text>
  <text x="260" y="154" text-anchor="middle" fill="#64748b" font-size="10" font-family="sans-serif">ReAct / Reflexion / VOYAGER</text>
  <line x1="260" y1="164" x2="260" y2="180" stroke="#334155" marker-end="url(#ar)"/>
  <rect fill="#dbeafe" stroke="#1d4ed8" x="120" y="184" width="280" height="44" rx="4"/>
  <text x="260" y="202" text-anchor="middle" fill="#0f172a" font-size="11" font-weight="bold" {F}>\u6838\u5fc3\u80fd\u529b</text>
  <text x="260" y="218" text-anchor="middle" fill="#64748b" font-size="10" {F}>\u8bb0\u5fc6 / \u89c4\u5212 / \u53cd\u601d / \u5de5\u5177</text>
  <line x1="260" y1="228" x2="260" y2="244" stroke="#334155" marker-end="url(#ar)"/>
  <rect fill="#fef3c7" stroke="#b45309" x="120" y="248" width="280" height="44" rx="4"/>
  <text x="260" y="266" text-anchor="middle" fill="#0f172a" font-size="11" font-weight="bold" {F}>\u57fa\u7840\u6a21\u578b</text>
  <text x="260" y="282" text-anchor="middle" fill="#64748b" font-size="10" font-family="sans-serif">GPT-4 / Claude / Llama</text>""",
    )
    (ROOT / "llm-game-agents-overview-tech-stack.svg").write_text(stack, encoding="utf-8", newline="\n")

    # --- \u91d1\u5b57\u5854 ---
    pyr = svg(
        620,
        380,
        MARK,
        f"""  <text x="310" y="26" text-anchor="middle" fill="#0f172a" font-size="13" font-weight="bold" {F}>\u6280\u672f\u5c42\u6b21\u91d1\u5b57\u5854</text>
  <rect fill="#fce7f3" stroke="#9d174d" x="210" y="40" width="200" height="64" rx="4"/>
  <text x="310" y="62" text-anchor="middle" fill="#0f172a" font-size="11" font-weight="bold" {F}>\u4e0a\u5c42\u5e94\u7528</text>
  <text x="310" y="78" text-anchor="middle" fill="#475569" font-size="9" {F}>\u7ade\u6280 / \u793e\u4ea4 / \u7279\u5b9a\u6e38\u620f</text>
  <text x="310" y="94" text-anchor="middle" fill="#64748b" font-size="8" font-family="sans-serif">Werewolf, Poker, StarCraft\u2026</text>
  <line x1="310" y1="104" x2="310" y2="124" stroke="#334155" marker-end="url(#ar)"/>
  <rect fill="#fff" stroke="#334155" x="40" y="128" width="170" height="72" rx="4"/>
  <text x="125" y="148" text-anchor="middle" fill="#0f172a" font-size="10" font-weight="bold" {F}>\u4e2d\u95f4\u5c42 \xb7 \u73af\u5883\u9002\u914d</text>
  <text x="125" y="164" text-anchor="middle" fill="#64748b" font-size="8" {F}>Crafter, Minecraft</text>
  <rect fill="#fff" stroke="#334155" x="225" y="128" width="170" height="72" rx="4"/>
  <text x="310" y="150" text-anchor="middle" fill="#0f172a" font-size="10" font-weight="bold" {F}>\u6a21\u62df\u4eff\u771f</text>
  <text x="310" y="168" text-anchor="middle" fill="#64748b" font-size="9" font-family="sans-serif">Generative Agents</text>
  <rect fill="#fff" stroke="#334155" x="410" y="128" width="170" height="72" rx="4"/>
  <text x="495" y="148" text-anchor="middle" fill="#0f172a" font-size="10" font-weight="bold" {F}>\u591a\u667a\u80fd\u4f53\u534f\u4f5c</text>
  <line x1="125" y1="128" x2="280" y2="50" stroke="#cbd5e1"/>
  <line x1="310" y1="128" x2="310" y2="104" stroke="#cbd5e1"/>
  <line x1="495" y1="128" x2="340" y2="50" stroke="#cbd5e1"/>
  <line x1="125" y1="200" x2="310" y2="240" stroke="#334155" marker-end="url(#ar)"/>
  <line x1="310" y1="200" x2="310" y2="240" stroke="#334155" marker-end="url(#ar)"/>
  <line x1="495" y1="200" x2="310" y2="240" stroke="#334155" marker-end="url(#ar)"/>
  <rect fill="#e2e8f0" stroke="#475569" x="180" y="244" width="260" height="112" rx="4"/>
  <text x="310" y="266" text-anchor="middle" fill="#0f172a" font-size="11" font-weight="bold" {F}>\u57fa\u7840\u6846\u67b6\u5c42</text>
  <text x="310" y="288" text-anchor="middle" fill="#475569" font-size="9" font-family="sans-serif">\u2022 ReAct \u2014 \u63a8\u7406+\u884c\u52a8</text>
  <text x="310" y="306" text-anchor="middle" fill="#475569" font-size="9" font-family="sans-serif">\u2022 Reflexion \u2014 \u81ea\u6211\u53cd\u601d</text>
  <text x="310" y="324" text-anchor="middle" fill="#475569" font-size="9" font-family="sans-serif">\u2022 Grounding RL \u2014 \u73af\u5883\u4ea4\u4e92\u5b66\u4e60</text>""",
    )
    (ROOT / "llm-game-agents-overview-pyramid.svg").write_text(pyr, encoding="utf-8", newline="\n")

    # --- NLP LLM \u6f14\u8fdb ---
    nlp = svg(
        480,
        220,
        MARK,
        f"""  <text x="240" y="22" text-anchor="middle" fill="#0f172a" font-size="12" font-weight="bold" {F}>LLM \u67b6\u6784\u6f14\u8fdb\uff08\u5bb6\u65cf\uff09</text>
  <text x="240" y="48" text-anchor="middle" fill="#475569" font-size="10" font-family="monospace">GPT-1\u2192 GPT-2\u2192 GPT-3\u2192 ChatGPT\u2192 GPT-4</text>
  <line x1="240" y1="54" x2="240" y2="72" stroke="#334155" marker-end="url(#ar)"/>
  <text x="240" y="90" text-anchor="middle" fill="#475569" font-size="10" font-family="monospace">BERT\u2192 RoBERTa\u2192 DeBERTa</text>
  <line x1="240" y1="96" x2="240" y2="114" stroke="#334155" marker-end="url(#ar)"/>
  <text x="240" y="132" text-anchor="middle" fill="#475569" font-size="10" font-family="monospace">T5\u2192 Flan-T5\u2192 UL2</text>
  <line x1="240" y1="138" x2="240" y2="156" stroke="#334155" marker-end="url(#ar)"/>
  <text x="240" y="174" text-anchor="middle" fill="#475569" font-size="10" font-family="monospace">LLaMA\u2192 LLaMA 2\u2192 Mistral\u2192 Mixtral</text>""",
    )
    (ROOT / "nlp-llm-evolution-families.svg").write_text(nlp, encoding="utf-8", newline="\n")

    # --- MRC \u65f6\u95f4\u7ebf ---
    mrc = svg(
        540,
        280,
        "",
        f"""  <text x="270" y="22" text-anchor="middle" fill="#0f172a" font-size="12" font-weight="bold" {F}>\u795e\u7ecf\u7f51\u7edc MRC \u53d1\u5c55\u65f6\u95f4\u7ebf</text>
  <rect fill="#f8fafc" stroke="#334155" x="16" y="36" width="508" height="40" rx="3"/>
  <text x="24" y="56" fill="#1e293b" font-size="10" font-weight="bold" {F}>2015-2016</text>
  <text x="24" y="70" fill="#475569" font-size="9" {F}>Attentive Reader, Impatient Reader, BiDAF</text>
  <rect fill="#f1f5f9" stroke="#334155" x="16" y="84" width="508" height="40" rx="3"/>
  <text x="24" y="104" fill="#1e293b" font-size="10" font-weight="bold" {F}>2017-2018</text>
  <text x="24" y="118" fill="#475569" font-size="9" {F}>R-Net, QANet, BERT</text>
  <rect fill="#e2e8f0" stroke="#334155" x="16" y="132" width="508" height="40" rx="3"/>
  <text x="24" y="152" fill="#1e293b" font-size="10" font-weight="bold" {F}>2019-2020</text>
  <text x="24" y="166" fill="#475569" font-size="9" {F}>RoBERTa, ALBERT, T5</text>
  <rect fill="#dbeafe" stroke="#1d4ed8" x="16" y="180" width="508" height="40" rx="3"/>
  <text x="24" y="200" fill="#1e293b" font-size="10" font-weight="bold" {F}>2021-2023</text>
  <text x="24" y="214" fill="#475569" font-size="9" {F}>GPT-3, ChatGPT, GPT-4, LLaMA</text>
  <rect fill="#fef3c7" stroke="#b45309" x="16" y="228" width="508" height="40" rx="3"/>
  <text x="24" y="248" fill="#1e293b" font-size="10" font-weight="bold" {F}>2024-</text>
  <text x="24" y="262" fill="#475569" font-size="9" {F}>RAG, Vision-Language Models</text>""",
    )
    (ROOT / "mrc-neural-timeline.svg").write_text(mrc, encoding="utf-8", newline="\n")

    dag_clean = svg(
        580,
        110,
        MARK,
        f"""  <text x="290" y="16" text-anchor="middle" fill="#0f172a" font-size="11" font-weight="bold" {F}>\u56e0\u679c\u56fe\uff1a\u94fe\u5f0f / \u6df7\u6742 / \u5bf9\u649e</text>
  <text x="55" y="62" text-anchor="middle" font-size="12" fill="#0f172a">X</text>
  <line x1="68" y1="58" x2="86" y2="58" stroke="#334155" stroke-width="1.5" marker-end="url(#ar)"/>
  <text x="105" y="62" text-anchor="middle" font-size="12" fill="#0f172a">Y</text>
  <line x1="118" y1="58" x2="136" y2="58" stroke="#334155" stroke-width="1.5" marker-end="url(#ar)"/>
  <text x="155" y="62" text-anchor="middle" font-size="12" fill="#0f172a">Z</text>
  <text x="105" y="88" text-anchor="middle" fill="#64748b" font-size="9" {F}>\u94fe\u5f0f</text>
  <text x="275" y="38" text-anchor="middle" font-size="12" fill="#0f172a">W</text>
  <line x1="265" y1="46" x2="246" y2="68" stroke="#334155" stroke-width="1.5" marker-end="url(#ar)"/>
  <line x1="285" y1="46" x2="304" y2="68" stroke="#334155" stroke-width="1.5" marker-end="url(#ar)"/>
  <text x="235" y="78" text-anchor="middle" font-size="12" fill="#0f172a">X</text>
  <text x="315" y="78" text-anchor="middle" font-size="12" fill="#0f172a">Y</text>
  <text x="275" y="96" text-anchor="middle" fill="#64748b" font-size="9" {F}>\u6df7\u6742</text>
  <text x="430" y="78" text-anchor="middle" font-size="12" fill="#0f172a">X</text>
  <text x="490" y="78" text-anchor="middle" font-size="12" fill="#0f172a">Y</text>
  <line x1="436" y1="68" x2="452" y2="42" stroke="#334155" stroke-width="1.5" marker-end="url(#ar)"/>
  <line x1="484" y1="68" x2="468" y2="42" stroke="#334155" stroke-width="1.5" marker-end="url(#ar)"/>
  <text x="460" y="38" text-anchor="middle" font-size="12" fill="#0f172a">W</text>
  <text x="460" y="96" text-anchor="middle" fill="#64748b" font-size="9" {F}>\u5bf9\u649e</text>""",
    )
    (ROOT / "causal-graph-three-patterns.svg").write_text(dag_clean, encoding="utf-8", newline="\n")

    # \u603b\u89c8 \u7406\u60f3\u67b6\u6784\u516c\u5f0f
    ideal = svg(
        520,
        120,
        "",
        f"""  <text x="260" y="22" text-anchor="middle" fill="#0f172a" font-size="12" font-weight="bold" {F}>\u7406\u60f3\u7ec4\u5408\u67b6\u6784</text>
  <text x="260" y="46" text-anchor="middle" fill="#475569" font-size="11" {F}>= Generative Agents \u8bb0\u5fc6\u6d41</text>
  <text x="260" y="64" text-anchor="middle" fill="#475569" font-size="11" {F}>+ VOYAGER \u6280\u80fd\u5e93</text>
  <text x="260" y="82" text-anchor="middle" fill="#475569" font-size="11" {F}>+ Reflexion \u5931\u8d25\u53cd\u601d</text>""",
    )
    (ROOT / "llm-game-agents-overview-ideal-trio.svg").write_text(ideal, encoding="utf-8", newline="\n")

    print("wrote diagrams to", ROOT)


if __name__ == "__main__":
    main()
