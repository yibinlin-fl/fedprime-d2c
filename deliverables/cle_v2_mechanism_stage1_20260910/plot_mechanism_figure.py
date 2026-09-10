"""Generate the dependency-free vector source for the Stage-1 paper figure."""

from pathlib import Path


OUT = Path(__file__).resolve().parent
TOP, BOTTOM = 105, 465
PLOT_HEIGHT = BOTTOM - TOP
HFL_COLOR, LOCAL_COLOR = "#3568B8", "#E58B3A"


def left_y(value: float) -> float:
    return BOTTOM - (value + 0.012) / 0.154 * PLOT_HEIGHT


def right_y(value: float) -> float:
    return BOTTOM - value / 0.142 * PLOT_HEIGHT


def make_bar(x: float, value: float, color: str, scale) -> str:
    zero, target = scale(0.0), scale(value)
    return (
        f'<rect x="{x:.1f}" y="{min(zero, target):.1f}" width="72" '
        f'height="{max(abs(zero-target), 1.2):.1f}" rx="2" fill="{color}"/>'
    )


svg = [
    '<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="600" viewBox="0 0 1400 600">',
    '<rect width="100%" height="100%" fill="white"/>',
    '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#202124}.title{font-size:27px;font-weight:700}.panel{font-size:18px;font-weight:700}.axis{font-size:15px}.small{font-size:13px;fill:#555}.value{font-size:15px;font-weight:700}.grid{stroke:#e5e7eb;stroke-width:1}.base{stroke:#333;stroke-width:1.3}</style>',
    '<text x="700" y="39" text-anchor="middle" class="title">CLE-HFL v2 mechanism evidence</text>',
    '<text x="80" y="78" class="panel">(a) Binding-aligned shortcut emerges under CLE</text>',
    '<text x="760" y="78" class="panel">(b) Most of the HFL effect is already local</text>',
]

for tick in (0.00, 0.04, 0.08, 0.12):
    y = left_y(tick)
    svg += [
        f'<line x1="80" y1="{y:.1f}" x2="670" y2="{y:.1f}" class="grid"/>',
        f'<text x="69" y="{y+5:.1f}" text-anchor="end" class="axis">{tick:.2f}</text>',
    ]
zero_left, null_y = left_y(0), left_y(0.0295586)
svg += [
    f'<line x1="80" y1="{zero_left:.1f}" x2="670" y2="{zero_left:.1f}" class="base"/>',
    f'<line x1="80" y1="{null_y:.1f}" x2="670" y2="{null_y:.1f}" stroke="#777" stroke-width="1.5" stroke-dasharray="7 6"/>',
    f'<text x="660" y="{null_y-8:.1f}" text-anchor="end" class="small">H9 shuffled null p95 = 0.030</text>',
    '<text transform="translate(22 290) rotate(-90)" text-anchor="middle" class="axis">Directional Shortcut Alignment (DSA)</text>',
]
for x, value, color in [
    (190, -0.0002450674, HFL_COLOR), (270, -0.0019143519, LOCAL_COLOR),
    (470, 0.1196442241, HFL_COLOR), (550, 0.1060461036, LOCAL_COLOR),
]:
    svg.append(make_bar(x, value, color, left_y))
    label_y = left_y(value) - 9 if value >= 0 else left_y(value) + 18
    svg.append(f'<text x="{x+36}" y="{label_y:.1f}" text-anchor="middle" class="value">{value:.3f}</text>')
svg += [
    '<text x="266" y="502" text-anchor="middle" class="axis">No CLE (gamma=0)</text>',
    '<text x="546" y="502" text-anchor="middle" class="axis">Strong CLE (gamma=0.9)</text>',
    f'<rect x="164" y="527" width="18" height="18" fill="{HFL_COLOR}"/><text x="191" y="541" class="axis">HFL</text>',
    f'<rect x="248" y="527" width="18" height="18" fill="{LOCAL_COLOR}"/><text x="275" y="541" class="axis">Local</text>',
]

for tick in (0.00, 0.04, 0.08, 0.12):
    y = right_y(tick)
    svg += [
        f'<line x1="770" y1="{y:.1f}" x2="1330" y2="{y:.1f}" class="grid"/>',
        f'<text x="759" y="{y+5:.1f}" text-anchor="end" class="axis">{tick:.2f}</text>',
    ]
svg += [
    f'<line x1="770" y1="{BOTTOM}" x2="1330" y2="{BOTTOM}" class="base"/>',
    '<text transform="translate(712 290) rotate(-90)" text-anchor="middle" class="axis">Delta DSA: gamma=0.9 - gamma=0</text>',
]
for x, value, low, high, color, label in [
    (875, 0.1198892916, 0.1180502157, 0.1215617399, HFL_COLOR, "HFL CLE effect"),
    (1095, 0.1079604555, 0.1061453441, 0.1096778702, LOCAL_COLOR, "Local CLE effect"),
]:
    y, center = right_y(value), x + 50
    svg += [
        f'<rect x="{x}" y="{y:.1f}" width="100" height="{BOTTOM-y:.1f}" rx="2" fill="{color}"/>',
        f'<line x1="{center}" y1="{right_y(high):.1f}" x2="{center}" y2="{right_y(low):.1f}" stroke="#222" stroke-width="2"/>',
        f'<line x1="{center-10}" y1="{right_y(high):.1f}" x2="{center+10}" y2="{right_y(high):.1f}" stroke="#222" stroke-width="2"/>',
        f'<line x1="{center-10}" y1="{right_y(low):.1f}" x2="{center+10}" y2="{right_y(low):.1f}" stroke="#222" stroke-width="2"/>',
        f'<text x="{center}" y="{y-13:.1f}" text-anchor="middle" class="value">{value:.3f}</text>',
        f'<text x="{center}" y="502" text-anchor="middle" class="axis">{label}</text>',
    ]
svg += [
    '<rect x="1160" y="313" width="165" height="72" rx="8" fill="white" stroke="#777"/>',
    '<text x="1242" y="340" text-anchor="middle" class="axis">Local/HFL = 90.05%</text>',
    '<text x="1242" y="368" text-anchor="middle" class="axis">Pooled add-on = 0.0119</text>',
    '<text x="700" y="582" text-anchor="middle" class="small">Fixed CLE-v2 scenario | training seed 0 | 12 rounds | 1,000 sources x 15 operators x 4 clients</text>',
    '</svg>',
]

(OUT / "MECHANISM_FIGURE.svg").write_text("\n".join(svg), encoding="utf-8")
