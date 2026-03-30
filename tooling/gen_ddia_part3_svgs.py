# -*- coding: utf-8 -*-
"""DDIA Part3 -> source/images/ddia/part3/. Run: python3 tooling/gen_ddia_part3_svgs.py"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "source" / "images" / "ddia" / "part3"
F = 'font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif"'
MARK = """  <defs>
    <marker id="ar" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><polygon points="0 0,8 3,0 6" fill="#334155"/></marker>
  </defs>"""


def write(name: str, body: str) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / name).write_text(body, encoding="utf-8", newline="\n")
    print(ROOT / name)


def svg(w: int, h: int, defs: str, inner: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img">\n'
        f"{defs}\n{inner}\n</svg>\n"
    )


def main() -> None:
    # 10.1
    write(
        "system-types-comparison.svg",
        svg(
            600,
            220,
            MARK,
            f"""  <rect fill="#f8fafc" stroke="#334155" stroke-width="1.5" x="8" y="8" width="584" height="204" rx="6"/>
  <text x="300" y="30" text-anchor="middle" fill="#0f172a" font-size="13" font-weight="bold" {F}>\u5728\u7ebf / \u6279\u5904\u7406 / \u6d41\u5904\u7406</text>
  <text x="24" y="58" fill="#0f172a" font-size="11" {F}>\u5728\u7ebf\u670d\u52a1\uff1a</text>
  <text x="104" y="58" fill="#475569" font-size="10" {F}>\u7528\u6237\u8bf7\u6c42</text>
  <line x1="160" y1="54" x2="188" y2="54" stroke="#334155" marker-end="url(#ar)"/>
  <rect fill="#e2e8f0" stroke="#475569" x="192" y="42" width="72" height="24" rx="3"/><text x="228" y="58" text-anchor="middle" fill="#0f172a" font-size="10" {F}>\u670d\u52a1</text>
  <line x1="264" y1="54" x2="292" y2="54" stroke="#334155" marker-end="url(#ar)"/>
  <text x="298" y="58" fill="#475569" font-size="10" {F}>\u54cd\u5e94\uff08\u6beb\u79d2\u7ea7\uff09</text>
  <text x="24" y="96" fill="#0f172a" font-size="11" {F}>\u6279\u5904\u7406\uff1a</text>
  <rect fill="#fff" stroke="#334155" x="104" y="78" width="120" height="36" rx="4"/>
  <text x="164" y="94" text-anchor="middle" fill="#0f172a" font-size="10" {F}>\u5927\u91cf\u8f93\u5165</text>
  <text x="164" y="106" text-anchor="middle" fill="#64748b" font-size="9" {F}>(TB)</text>
  <line x1="224" y1="96" x2="252" y2="96" stroke="#334155" marker-end="url(#ar)"/>
  <rect fill="#fff" stroke="#334155" x="256" y="78" width="120" height="36" rx="4"/>
  <text x="316" y="94" text-anchor="middle" fill="#0f172a" font-size="10" {F}>\u6279\u5904\u7406\u4f5c\u4e1a</text>
  <text x="316" y="106" text-anchor="middle" fill="#64748b" font-size="9" {F}>(\u8fd0\u884c\u6570\u5c0f\u65f6)</text>
  <line x1="376" y1="96" x2="404" y2="96" stroke="#334155" marker-end="url(#ar)"/>
  <text x="412" y="100" fill="#475569" font-size="10" {F}>\u8f93\u51fa\u7ed3\u679c</text>
  <text x="24" y="148" fill="#0f172a" font-size="11" {F}>\u6d41\u5904\u7406\uff1a</text>
  <text x="104" y="148" fill="#475569" font-size="10" {F}>\u4e8b\u4ef6\u6d41</text>
  <line x1="152" y1="144" x2="180" y2="144" stroke="#334155" marker-end="url(#ar)"/>
  <rect fill="#dbeafe" stroke="#1d4ed8" x="184" y="132" width="72" height="24" rx="3"/><text x="220" y="148" text-anchor="middle" fill="#0f172a" font-size="10" {F}>\u5904\u7406</text>
  <line x1="256" y1="144" x2="284" y2="144" stroke="#334155" marker-end="url(#ar)"/>
  <text x="290" y="148" fill="#475569" font-size="10" {F}>\u8f93\u51fa\u6d41\uff08\u6301\u7eed\uff09</text>""",
        ),
    )

    # MapReduce
    rows = []
    for i in range(3):
        y = 64 + i * 40
        rows.append(
            f'<rect x="28" y="{y}" width="52" height="28" rx="3" fill="#fff" stroke="#334155"/>'
            f'<text x="54" y="{y + 18}" text-anchor="middle" fill="#0f172a" font-size="9" {F}>{i + 1}</text>'
            f'<rect x="122" y="{y}" width="64" height="28" rx="3" fill="#e2e8f0" stroke="#475569"/>'
            f'<text x="154" y="{y + 18}" text-anchor="middle" fill="#0f172a" font-size="9" {F}>Map {i + 1}</text>'
        )
    body_mr = (
        f"""  <text x="280" y="22" text-anchor="middle" fill="#0f172a" font-size="13" font-weight="bold" {F}>MapReduce \u6d41\u7a0b</text>
  <text x="48" y="52" fill="#64748b" font-size="9" {F}>\u5206\u7247</text>
  {''.join(rows)}
  <line x1="80" y1="78" x2="118" y2="78" stroke="#334155" marker-end="url(#ar)"/>
  <line x1="80" y1="118" x2="118" y2="118" stroke="#334155" marker-end="url(#ar)"/>
  <line x1="80" y1="158" x2="118" y2="158" stroke="#334155" marker-end="url(#ar)"/>
  <line x1="186" y1="78" x2="230" y2="100" stroke="#334155" marker-end="url(#ar)"/>
  <line x1="186" y1="118" x2="248" y2="108" stroke="#334155" marker-end="url(#ar)"/>
  <line x1="186" y1="158" x2="230" y2="140" stroke="#334155" marker-end="url(#ar)"/>
  <rect fill="#fef3c7" stroke="#b45309" x="248" y="88" width="88" height="56" rx="4"/>
  <text x="292" y="108" text-anchor="middle" fill="#0f172a" font-size="10" {F}>\u6309\u952e\u5206\u7ec4</text>
  <text x="292" y="126" text-anchor="middle" fill="#64748b" font-size="9" font-family="sans-serif">Shuffle</text>
  <line x1="336" y1="108" x2="380" y2="88" stroke="#334155" marker-end="url(#ar)"/>
  <line x1="336" y1="116" x2="380" y2="138" stroke="#334155" marker-end="url(#ar)"/>
  <rect fill="#dcfce7" stroke="#15803d" x="384" y="72" width="80" height="32" rx="3"/><text x="424" y="92" text-anchor="middle" fill="#0f172a" font-size="10" font-family="sans-serif">Reduce 1</text>
  <rect fill="#dcfce7" stroke="#15803d" x="384" y="124" width="80" height="32" rx="3"/><text x="424" y="144" text-anchor="middle" fill="#0f172a" font-size="10" font-family="sans-serif">Reduce 2</text>
  <line x1="464" y1="88" x2="498" y2="88" stroke="#334155" marker-end="url(#ar)"/>
  <line x1="464" y1="140" x2="498" y2="140" stroke="#334155" marker-end="url(#ar)"/>
  <text x="530" y="92" fill="#475569" font-size="9" {F}>\u7ed3\u679c1</text>
  <text x="530" y="144" fill="#475569" font-size="9" {F}>\u7ed3\u679c2</text>
  <text x="280" y="188" text-anchor="middle" fill="#64748b" font-size="9" {F}>\u8f93\u5165 \u2192 Map \u2192 Shuffle \u2192 Reduce \u2192 \u8f93\u51fa</text>"""
    )
    write("mapreduce-flow.svg", svg(560, 200, MARK, body_mr))

    write(
        "message-queue-traditional.svg",
        svg(
            440,
            165,
            MARK,
            f"""  <rect fill="#f8fafc" stroke="#334155" stroke-width="2" x="60" y="24" width="320" height="56" rx="4"/>
  <text x="220" y="44" text-anchor="middle" fill="#1e293b" font-size="11" font-weight="bold" {F}>\u961f\u5217</text>
  <text x="90" y="62" fill="#0f172a" font-size="10" font-family="monospace">msg1</text>
  <text x="150" y="62" fill="#0f172a" font-size="10" font-family="monospace">msg2</text>
  <text x="210" y="62" fill="#0f172a" font-size="10" font-family="monospace">msg3</text>
  <text x="270" y="62" fill="#0f172a" font-size="10" font-family="monospace">msg4</text>
  <line x1="100" y1="80" x2="100" y2="100" stroke="#334155" marker-end="url(#ar)"/>
  <line x1="160" y1="80" x2="160" y2="100" stroke="#334155" marker-end="url(#ar)"/>
  <line x1="220" y1="80" x2="220" y2="100" stroke="#334155" marker-end="url(#ar)"/>
  <line x1="280" y1="80" x2="280" y2="100" stroke="#334155" marker-end="url(#ar)"/>
  <text x="100" y="120" text-anchor="middle" fill="#475569" font-size="9" {F}>\u6d88\u8d391</text>
  <text x="160" y="120" text-anchor="middle" fill="#475569" font-size="9" {F}>\u6d88\u8d392</text>
  <text x="220" y="120" text-anchor="middle" fill="#475569" font-size="9" {F}>\u6d88\u8d391</text>
  <text x="280" y="120" text-anchor="middle" fill="#475569" font-size="9" {F}>\u6d88\u8d392</text>
  <text x="220" y="150" text-anchor="middle" fill="#64748b" font-size="10" {F}>\u6bcf\u6761\u6d88\u606f\u4ec5\u88ab\u4e00\u4e2a\u6d88\u8d39\u8005\u5904\u7406\uff1b\u6d88\u606f\u88ab\u5220\u9664</text>""",
        ),
    )

    write(
        "kafka-partitions-consumers.svg",
        svg(
            600,
            170,
            MARK,
            f"""  <text x="300" y="22" text-anchor="middle" fill="#0f172a" font-size="12" font-weight="bold" {F}>Kafka \u5206\u533a\u4e0e\u6d88\u8d39\u8005</text>
  <text x="24" y="52" fill="#0f172a" font-size="10" {F}>\u5206\u533a0:</text>
  <rect fill="#e2e8f0" stroke="#64748b" x="80" y="40" width="360" height="22" rx="2"/>
  <text x="260" y="55" text-anchor="middle" fill="#475569" font-size="9" font-family="monospace">[msg0][msg3][msg6]\u2026</text>
  <text x="456" y="55" fill="#15803d" font-size="9" {F}>\u2192 A</text>
  <text x="24" y="84" fill="#0f172a" font-size="10" {F}>\u5206\u533a1:</text>
  <rect fill="#e2e8f0" stroke="#64748b" x="80" y="72" width="360" height="22" rx="2"/>
  <text x="260" y="87" text-anchor="middle" fill="#475569" font-size="9" font-family="monospace">[msg1][msg4][msg7]\u2026</text>
  <text x="456" y="87" fill="#15803d" font-size="9" {F}>\u2192 B</text>
  <text x="24" y="116" fill="#0f172a" font-size="10" {F}>\u5206\u533a2:</text>
  <rect fill="#e2e8f0" stroke="#64748b" x="80" y="104" width="360" height="22" rx="2"/>
  <text x="260" y="119" text-anchor="middle" fill="#475569" font-size="9" font-family="monospace">[msg2][msg5][msg8]\u2026</text>
  <text x="456" y="119" fill="#15803d" font-size="9" {F}>\u2192 C</text>
  <text x="300" y="158" text-anchor="middle" fill="#64748b" font-size="9" {F}>\u6301\u4e45\u5316 \xb7 \u5206\u533a\u5185\u6709\u5e8f \xb7 \u591a\u6d88\u8d39\u8005\u7ec4</text>""",
        ),
    )

    kc_fixed = (
        svg(520, 250, MARK, f"""  <rect fill="#e2e8f0" stroke="#334155" stroke-width="2" x="40" y="12" width="440" height="118" rx="6"/>
  <text x="260" y="32" text-anchor="middle" fill="#0f172a" font-size="12" font-weight="bold" {F}>Kafka Cluster \xb7 Topic: orders</text>
  <rect fill="#fff" stroke="#64748b" x="56" y="44" width="188" height="72" rx="4"/>
  <text x="150" y="62" text-anchor="middle" fill="#1e293b" font-size="10" font-weight="bold" {F}>Partition 0</text>
  <text x="150" y="76" text-anchor="middle" fill="#475569" font-size="9" font-family="monospace">[0][1][2]\u2026</text>
  <text x="150" y="92" text-anchor="middle" fill="#64748b" font-size="8" {F}>Leader: B1 Replica: B2</text>
  <rect fill="#fff" stroke="#64748b" x="276" y="44" width="188" height="72" rx="4"/>
  <text x="370" y="62" text-anchor="middle" fill="#1e293b" font-size="10" font-weight="bold" {F}>Partition 1</text>
  <text x="370" y="76" text-anchor="middle" fill="#475569" font-size="9" font-family="monospace">[0][1][2]\u2026</text>
  <text x="370" y="92" text-anchor="middle" fill="#64748b" font-size="8" {F}>Leader: B2 Replica: B1</text>
  <text x="260" y="112" text-anchor="middle" fill="#64748b" font-size="9" {F}>Broker 1 / Broker 2</text>
  <line x1="150" y1="130" x2="120" y2="168" stroke="#334155" marker-end="url(#ar)"/>
  <line x1="260" y1="130" x2="260" y2="168" stroke="#334155" marker-end="url(#ar)"/>
  <line x1="370" y1="130" x2="400" y2="168" stroke="#334155" marker-end="url(#ar)"/>
  <text x="120" y="188" text-anchor="middle" fill="#475569" font-size="9" {F}>C1 / Group A</text>
  <text x="260" y="188" text-anchor="middle" fill="#475569" font-size="9" {F}>C2 / Group A</text>
  <text x="400" y="188" text-anchor="middle" fill="#475569" font-size="9" {F}>C3 / Group B</text>""")
    )
    write("kafka-cluster-topic.svg", kc_fixed)

    write(
        "cdc-pipeline.svg",
        svg(
            520,
            200,
            MARK,
            f"""  <rect fill="#fff" stroke="#334155" x="24" y="24" width="88" height="36" rx="4"/><text x="68" y="46" text-anchor="middle" fill="#0f172a" font-size="10" {F}>\u5e94\u7528</text>
  <line x1="112" y1="42" x2="138" y2="42" stroke="#334155" marker-end="url(#ar)"/>
  <rect fill="#fff" stroke="#334155" x="142" y="24" width="88" height="36" rx="4"/><text x="186" y="46" text-anchor="middle" fill="#0f172a" font-size="10" {F}>\u6570\u636e\u5e93</text>
  <rect fill="#fef3c7" stroke="#b45309" x="280" y="24" width="100" height="44" rx="4"/>
  <text x="330" y="42" text-anchor="middle" fill="#0f172a" font-size="10" {F}>CDC</text>
  <text x="330" y="56" text-anchor="middle" fill="#64748b" font-size="8" font-family="sans-serif">Debezium</text>
  <line x1="230" y1="42" x2="272" y2="42" stroke="#334155" marker-end="url(#ar)"/>
  <line x1="186" y1="60" x2="186" y2="100" stroke="#334155" marker-end="url(#ar)"/>
  <rect fill="#e2e8f0" stroke="#64748b" x="142" y="104" width="88" height="32" rx="3"/><text x="186" y="124" text-anchor="middle" fill="#475569" font-size="9" font-family="sans-serif">binlog</text>
  <text x="248" y="115" fill="#64748b" font-size="8" {F}>\u53d8\u66f4\u65e5\u5fd7</text>
  <line x1="330" y1="68" x2="330" y2="104" stroke="#334155" marker-end="url(#ar)"/>
  <rect fill="#dcfce7" stroke="#15803d" x="286" y="108" width="88" height="36" rx="4"/><text x="330" y="132" text-anchor="middle" fill="#0f172a" font-size="10" font-family="sans-serif">Kafka</text>
  <text x="400" y="90" fill="#64748b" font-size="9" {F}>\u4e8b\u4ef6\u6d41</text>""",
        ),
    )

    write(
        "event-sourcing-vs-traditional.svg",
        svg(
            560,
            175,
            MARK,
            f"""  <text x="140" y="24" text-anchor="middle" fill="#1e293b" font-size="11" font-weight="bold" {F}>\u4f20\u7edf\uff1a\u76f4\u63a5\u6539\u72b6\u6001</text>
  <text x="420" y="24" text-anchor="middle" fill="#1e293b" font-size="11" font-weight="bold" {F}>\u4e8b\u4ef6\u6eaf\u6e90</text>
  <text x="24" y="48" fill="#475569" font-size="10" {F}>\u4f59\u989d:</text>
  <text x="80" y="48" fill="#0f172a" font-size="10" font-family="monospace">100 \u2192 90 \u2192 140 \u2192 110</text>
  <text x="300" y="52" fill="#475569" font-size="10" {F}>\u4e8b\u4ef6\u65e5\u5fd7\uff1a</text>
  <text x="300" y="68" fill="#0f172a" font-size="9" {F}>1. \u521d\u59cb\u5b58\u6b3e 100</text>
  <text x="300" y="84" fill="#0f172a" font-size="9" {F}>2. \u53d6\u6b3e 10</text>
  <text x="300" y="100" fill="#0f172a" font-size="9" {F}>3. \u5b58\u6b3e 50</text>
  <text x="300" y="116" fill="#0f172a" font-size="9" {F}>4. \u53d6\u6b3e 30</text>
  <line x1="180" y1="130" x2="360" y2="130" stroke="#cbd5e1"/>
  <text x="280" y="155" text-anchor="middle" fill="#64748b" font-size="10" {F}>\u91cd\u653e\u4e8b\u4ef6 \u2192 \u8ba1\u7b97\u5f53\u524d\u72b6\u6001</text>""",
        ),
    )

    write(
        "stream-windows.svg",
        svg(
            520,
            125,
            "",
            f"""  <text x="16" y="22" fill="#1e293b" font-size="11" font-weight="bold" {F}>\u6eda\u52a8\u7a97\u53e3 (5\u5206\u949f)</text>
  <text x="16" y="42" fill="#475569" font-size="10" font-family="monospace">[00:00-05:00] [05:00-10:00] [10:00-15:00]</text>
  <text x="16" y="66" fill="#1e293b" font-size="11" font-weight="bold" {F}>\u6ed1\u52a8 (5\u5206\u949f, 1\u5206\u949f\u6b65\u957f)</text>
  <text x="16" y="86" fill="#475569" font-size="10" font-family="monospace">[00:00-05:00]</text>
  <text x="28" y="100" fill="#475569" font-size="10" font-family="monospace">[01:00-06:00]</text>
  <text x="40" y="114" fill="#475569" font-size="10" font-family="monospace">[02:00-07:00]</text>""",
        ),
    )

    write(
        "data-integration-arch.svg",
        svg(
            480,
            320,
            MARK,
            f"""  <rect fill="#f1f5f9" stroke="#334155" stroke-width="2" x="12" y="12" width="456" height="296" rx="6"/>
  <text x="240" y="34" text-anchor="middle" fill="#0f172a" font-size="13" font-weight="bold" {F}>\u6570\u636e\u96c6\u6210\u67b6\u6784</text>
  <rect fill="#fff" stroke="#334155" x="48" y="52" width="96" height="36" rx="3"/><text x="96" y="74" text-anchor="middle" fill="#0f172a" font-size="10" {F}>\u6570\u636e\u5e93</text>
  <rect fill="#fff" stroke="#334155" x="192" y="52" width="96" height="36" rx="3"/><text x="240" y="74" text-anchor="middle" fill="#0f172a" font-size="10" {F}>\u7f13\u5b58</text>
  <rect fill="#fff" stroke="#334155" x="336" y="52" width="96" height="36" rx="3"/><text x="384" y="74" text-anchor="middle" fill="#0f172a" font-size="10" {F}>\u641c\u7d22</text>
  <line x1="96" y1="88" x2="240" y2="120" stroke="#94a3b8"/>
  <line x1="240" y1="88" x2="240" y2="120" stroke="#94a3b8"/>
  <line x1="384" y1="88" x2="240" y2="120" stroke="#94a3b8"/>
  <rect fill="#fef3c7" stroke="#b45309" x="168" y="124" width="144" height="40" rx="4"/>
  <text x="240" y="142" text-anchor="middle" fill="#0f172a" font-size="11" font-weight="bold" {F}>\u4e8b\u4ef6\u65e5\u5fd7</text>
  <text x="240" y="156" text-anchor="middle" fill="#64748b" font-size="9" font-family="sans-serif">Kafka</text>
  <line x1="240" y1="164" x2="240" y2="188" stroke="#334155" marker-end="url(#ar)"/>
  <line x1="168" y1="180" x2="96" y2="220" stroke="#94a3b8" marker-end="url(#ar)"/>
  <line x1="240" y1="188" x2="240" y2="220" stroke="#94a3b8" marker-end="url(#ar)"/>
  <line x1="312" y1="180" x2="384" y2="220" stroke="#94a3b8" marker-end="url(#ar)"/>
  <rect fill="#fff" stroke="#334155" x="48" y="224" width="96" height="36" rx="3"/><text x="96" y="246" text-anchor="middle" fill="#0f172a" font-size="10" {F}>\u5206\u6790</text>
  <rect fill="#fff" stroke="#334155" x="192" y="224" width="96" height="36" rx="3"/><text x="240" y="246" text-anchor="middle" fill="#0f172a" font-size="10" {F}>ML\u5e73\u53f0</text>
  <rect fill="#fff" stroke="#334155" x="336" y="224" width="96" height="36" rx="3"/><text x="384" y="246" text-anchor="middle" fill="#0f172a" font-size="10" {F}>\u76d1\u63a7</text>""",
        ),
    )

    write(
        "lambda-architecture.svg",
        svg(
            380,
            300,
            MARK,
            f"""  <text x="190" y="22" text-anchor="middle" fill="#0f172a" font-size="12" font-weight="bold" {F}>Lambda \u67b6\u6784</text>
  <text x="190" y="44" text-anchor="middle" fill="#64748b" font-size="10" {F}>\u8f93\u5165\u6570\u636e</text>
  <line x1="190" y1="48" x2="120" y2="72" stroke="#334155" marker-end="url(#ar)"/>
  <line x1="190" y1="48" x2="260" y2="72" stroke="#334155" marker-end="url(#ar)"/>
  <rect fill="#e2e8f0" stroke="#334155" x="48" y="76" width="144" height="56" rx="4"/>
  <text x="120" y="96" text-anchor="middle" fill="#0f172a" font-size="10" font-weight="bold" {F}>\u6279\u5904\u7406\u5c42</text>
  <text x="120" y="112" text-anchor="middle" fill="#64748b" font-size="9" {F}>\u5168\u91cf</text>
  <text x="120" y="124" text-anchor="middle" fill="#64748b" font-size="8" font-family="sans-serif">MapReduce</text>
  <rect fill="#dbeafe" stroke="#1d4ed8" x="212" y="76" width="144" height="56" rx="4"/>
  <text x="284" y="96" text-anchor="middle" fill="#0f172a" font-size="10" font-weight="bold" {F}>\u901f\u5ea6\u5c42</text>
  <text x="284" y="112" text-anchor="middle" fill="#64748b" font-size="9" {F}>\u589e\u91cf</text>
  <text x="284" y="124" text-anchor="middle" fill="#64748b" font-size="8" font-family="sans-serif">Storm/Flink</text>
  <line x1="120" y1="132" x2="120" y2="152" stroke="#334155" marker-end="url(#ar)"/>
  <line x1="284" y1="132" x2="284" y2="152" stroke="#334155" marker-end="url(#ar)"/>
  <rect fill="#fff" stroke="#64748b" x="48" y="156" width="144" height="32" rx="3"/><text x="120" y="176" text-anchor="middle" fill="#475569" font-size="10" {F}>\u6279\u5904\u7406\u89c6\u56fe</text>
  <rect fill="#fff" stroke="#64748b" x="212" y="156" width="144" height="32" rx="3"/><text x="284" y="176" text-anchor="middle" fill="#475569" font-size="10" {F}>\u5b9e\u65f6\u89c6\u56fe</text>
  <line x1="120" y1="188" x2="190" y2="216" stroke="#334155" marker-end="url(#ar)"/>
  <line x1="284" y1="188" x2="190" y2="216" stroke="#334155" marker-end="url(#ar)"/>
  <rect fill="#fef3c7" stroke="#b45309" x="118" y="220" width="144" height="44" rx="4"/>
  <text x="190" y="240" text-anchor="middle" fill="#0f172a" font-size="11" font-weight="bold" {F}>\u670d\u52a1\u5c42</text>
  <text x="190" y="256" text-anchor="middle" fill="#64748b" font-size="9" {F}>\u5408\u5e76\u7ed3\u679c</text>""",
        ),
    )

    write(
        "kappa-architecture.svg",
        svg(
            320,
            280,
            MARK,
            f"""  <text x="160" y="24" text-anchor="middle" fill="#0f172a" font-size="12" font-weight="bold" {F}>Kappa \u67b6\u6784</text>
  <text x="160" y="48" text-anchor="middle" fill="#64748b" font-size="10" {F}>\u8f93\u5165\u6570\u636e</text>
  <line x1="160" y1="52" x2="160" y2="72" stroke="#334155" marker-end="url(#ar)"/>
  <rect fill="#fef3c7" stroke="#b45309" x="80" y="76" width="160" height="44" rx="4"/>
  <text x="160" y="96" text-anchor="middle" fill="#0f172a" font-size="10" font-weight="bold" {F}>\u65e5\u5fd7\u5b58\u50a8</text>
  <text x="160" y="112" text-anchor="middle" fill="#64748b" font-size="9" font-family="sans-serif">Kafka</text>
  <line x1="160" y1="120" x2="160" y2="144" stroke="#334155" marker-end="url(#ar)"/>
  <rect fill="#dbeafe" stroke="#1d4ed8" x="80" y="148" width="160" height="44" rx="4"/>
  <text x="160" y="168" text-anchor="middle" fill="#0f172a" font-size="10" font-weight="bold" {F}>\u6d41\u5904\u7406</text>
  <text x="160" y="184" text-anchor="middle" fill="#64748b" font-size="9" font-family="sans-serif">Flink</text>
  <line x1="160" y1="192" x2="160" y2="216" stroke="#334155" marker-end="url(#ar)"/>
  <rect fill="#fff" stroke="#334155" x="80" y="220" width="160" height="36" rx="4"/>
  <text x="160" y="242" text-anchor="middle" fill="#0f172a" font-size="11" font-weight="bold" {F}>\u670d\u52a1\u5c42</text>
  <text x="160" y="266" text-anchor="middle" fill="#64748b" font-size="9" {F}>\u91cd\u653e\u65e5\u5fd7\u5373\u53ef\u91cd\u7b97</text>""",
        ),
    )


if __name__ == "__main__":
    main()
