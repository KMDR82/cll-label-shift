"""Generates Figure_1 (system architecture) as an editable draw.io file."""
import sys
from xml.sax.saxutils import escape
out = sys.argv[1]
cells = []
BOX = ("rounded=1;arcSize=8;whiteSpace=wrap;html=1;fontFamily=Helvetica;fontSize=13;"
       "strokeColor=#333333;strokeWidth=1.2;fillColor=#FFFFFF;fontColor=#111111;spacing=4;")
DATA = BOX + "fillColor=#EEEEEE;"
CORE = BOX + "fillColor=#DCE8F5;strokeColor=#2F5F8F;"
OUT = BOX + "fillColor=#FFFFFF;strokeColor=#2F5F8F;strokeWidth=2;"
NOTE = ("rounded=0;whiteSpace=wrap;html=1;fontFamily=Helvetica;fontSize=12;dashed=1;"
        "strokeColor=#777777;fillColor=none;fontColor=#333333;align=left;spacingLeft=8;")
DIAM = ("rhombus;whiteSpace=wrap;html=1;fontFamily=Helvetica;fontSize=13;strokeColor=#2F5F8F;"
        "strokeWidth=1.2;fillColor=#FFFFFF;fontColor=#111111;")
LANE = "rounded=0;html=1;strokeColor=#BBBBBB;fillColor=#F8F8F8;"
LANET = ("text;html=1;fontFamily=Helvetica;fontSize=12;fontStyle=1;fontColor=#444444;"
         "horizontal=0;align=center;verticalAlign=middle;")
EDGE = ("edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;endSize=6;"
        "strokeColor=#333333;strokeWidth=1.2;fontFamily=Helvetica;fontSize=12;fontColor=#333333;"
        "labelBackgroundColor=#FFFFFF;")

def v(i, label, x, y, w, h, style):
    cells.append(f'<mxCell id="{i}" value="{escape(label, {chr(34): "&quot;"})}" style="{style}" '
                 f'vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')

def e(i, s, t, label="", ex=None, ey=None, nx=None, ny=None, pts=(), extra=""):
    st = EDGE + extra
    if ex is not None: st += f"exitX={ex};exitY={ey};exitDx=0;exitDy=0;"
    if nx is not None: st += f"entryX={nx};entryY={ny};entryDx=0;entryDy=0;"
    p = "".join(f'<mxPoint x="{a}" y="{b}"/>' for a, b in pts)
    pa = f'<Array as="points">{p}</Array>' if pts else ""
    cells.append(f'<mxCell id="{i}" value="{escape(label, {chr(34): "&quot;"})}" style="{st}" edge="1" parent="1" '
                 f'source="{s}" target="{t}"><mxGeometry relative="1" as="geometry">{pa}</mxGeometry></mxCell>')

# lanes
for i, (y, h, t) in enumerate([(10, 175, "Data and training"), (195, 190, "Label-free statistics"),
                               (395, 235, "Weight estimation")]):
    v(f"L{i}", "", 10, y, 810, h, LANE)
    v(f"LT{i}", t, 12, y, 26, h, LANET)

# band 1: data and predictor
v("pool", "Complementary-labelled pool<br>\\((x,\\bar y),\\ \\bar y\\sim Q[y,\\cdot]\\)<br>\\(y\\) not observed", 50, 22, 210, 78, DATA)
v("half", "Training half", 290, 34, 150, 54, BOX)
v("pred", "Predictor \\(h\\)<br>corrected risk with \\(Q^{-1}\\),<br>then fixed", 290, 112, 150, 62, CORE)
v("S", "Held-out split \\(S\\)<br>\\((x,\\bar y)\\) pairs", 50, 122, 210, 52, DATA)
v("tgt", "Unlabelled target sample<br>\\(p_{\\mathrm{te}}(x\\mid y)=p_{\\mathrm{tr}}(x\\mid y)\\)", 560, 112, 240, 62, DATA)

# band 2: statistics
v("Cb", "Complementary confusion<br>\\(\\hat{\\bar C}[i,j]=\\hat p_{\\mathrm{tr}}(h=i,\\bar y=j)\\)", 50, 215, 200, 66, CORE)
v("a", "Accuracy without labels<br>\\(\\hat a=1-(K-1)\\,\\mathrm{tr}\\,\\hat{\\bar C}\\)", 290, 215, 200, 66, CORE)
v("q", "Target prediction rates<br>\\(\\hat q_{\\mathrm{te}}[i]=\\hat p_{\\mathrm{te}}(h=i)\\)", 560, 215, 240, 66, CORE)
v("D", "Misfit diagnostic \\(\\hat D\\)<br>from two halves of \\(S\\)", 50, 303, 200, 66, CORE)

# band 3: estimation and selection
v("sel", "\\(\\hat D<\\tau\\,?\\)", 75, 455, 150, 100, DIAM)
v("clsc", "CL-SC<br>\\(\\Pi\\bigl((\\hat q_{\\mathrm{te}}-\\hat b\\mathbf 1)\\oslash((\\hat a-\\hat b)\\,p_{\\mathrm{tr}})\\bigr)\\)", 270, 418, 240, 66, BOX)
v("nv", "Naive<br>\\(\\Pi\\bigl(\\hat q_{\\mathrm{te}}\\oslash p_{\\mathrm{tr}}\\bigr)\\)", 270, 520, 240, 60, BOX)
v("out", "Importance weights \\(\\hat w\\)<br>corrected posterior<br>\\(\\hat p_{\\mathrm{te}}(k\\mid x)\\propto p(k\\mid x)\\,\\hat w_k\\)", 600, 455, 200, 100, OUT)

# edges
e("e1", "pool", "half", ex=1, ey=0.25, nx=0, ny=0.5)
e("e2", "pool", "S", ex=0.5, ey=1, nx=0.5, ny=0)
e("e3", "half", "pred", ex=0.5, ey=1, nx=0.5, ny=0)
e("e4", "S", "Cb", "\\(\\bar y\\)", ex=0.3, ey=1, nx=0.3, ny=0)
e("e5", "pred", "Cb", "\\(h(x)\\)", ex=0.3, ey=1, nx=0.8, ny=0, pts=[(335, 196), (210, 196)])
e("e6", "pred", "q", ex=0.7, ey=1, nx=0.3, ny=0, pts=[(395, 196), (632, 196)])
e("e7", "tgt", "q", "\\(x\\)", ex=0.7, ey=1, nx=0.7, ny=0)
e("e8", "Cb", "a", ex=1, ey=0.5, nx=0, ny=0.5)
e("e9", "Cb", "D", ex=0.5, ey=1, nx=0.5, ny=0)
e("e10", "D", "sel", "\\(\\hat D\\)", ex=0.5, ey=1, nx=0.5, ny=0, pts=[])
e("e11", "a", "clsc", "\\(\\hat a,\\ \\hat b\\)", ex=0.5, ey=1, nx=0.5, ny=0)
e("e12", "q", "clsc", "", ex=0.1, ey=1, nx=1, ny=0.4, pts=[(584, 445)])
e("e13", "q", "nv", ex=0.1, ey=1, nx=1, ny=0.5, pts=[(584, 550)])
e("e14", "clsc", "sel", "yes", ex=0, ey=0.5, nx=1, ny=0.5, pts=[(240, 451), (240, 505)])
e("e15", "nv", "sel", "no", ex=0, ey=0.5, nx=1, ny=0.5, pts=[(240, 550), (240, 505)])
e("e16", "sel", "out", "\\(\\hat w\\)", ex=0.5, ey=1, nx=0.5, ny=1, pts=[(150, 605), (700, 605)],
  extra="strokeColor=#2F5F8F;strokeWidth=1.6;")

xml = ('<mxfile host="drawio"><diagram name="Figure_1" id="arch">'
       '<mxGraphModel dx="800" dy="640" grid="0" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" '
       'fold="1" page="0" pageScale="1" pageWidth="830" pageHeight="640" math="1" shadow="0">'
       '<root><mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(cells) + '</root></mxGraphModel></diagram></mxfile>')
open(out, "w").write(xml)
print("written", out, len(cells), "cells")
