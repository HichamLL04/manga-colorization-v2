"""Build the printable SSD from docs/SSD.md with the bundled artifact runtime.

This script does not install or change project dependencies.
"""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
import re
import textwrap

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, KeepTogether, LongTable, PageBreak, PageTemplate, Paragraph,
    Preformatted, Spacer, TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/SSD.md"
DEST = ROOT / "output/pdf/SSD_MangaColor_v1.pdf"
QA = ROOT / "tmp/pdfs/ssd"


def font_setup():
    fonts = Path("C:/Windows/Fonts")
    for name, filename in [("SSD", "arial.ttf"), ("SSD-Bold", "arialbd.ttf"),
                           ("SSD-Italic", "ariali.ttf"), ("SSD-Mono", "consola.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(fonts / filename)))
    pdfmetrics.registerFontFamily("SSD", normal="SSD", bold="SSD-Bold",
                                  italic="SSD-Italic", boldItalic="SSD-Bold")


def inline(value):
    value = html.escape(value, quote=False)
    value = re.sub(r"`([^`]+)`", r'<font name="SSD-Mono">\1</font>', value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"(https?://[^\s<]+)", lambda m: '<link href="' + m[0] + '">' + m[0] + '</link>', value)
    return value


class SSDDocument(BaseDocTemplate):
    def __init__(self, filename):
        super().__init__(str(filename), pagesize=A4, leftMargin=46, rightMargin=46,
                         topMargin=48, bottomMargin=46,
                         title="Especificación del diseño de software de MangaColor",
                         author="Proyecto MangaColor", subject="SSD versión 1.0")
        self.addPageTemplates(PageTemplate(id="normal", frames=[
            Frame(self.leftMargin, self.bottomMargin, self.width, self.height,
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], onPage=self.decorate))

    def decorate(self, canvas, doc):
        canvas.saveState()
        canvas.setFont("SSD", 8)
        canvas.setFillColor(colors.HexColor("#58616d"))
        canvas.drawString(doc.leftMargin, 25, "MangaColor  |  SSD 1.0  |  16 noviembre 2026")
        canvas.drawRightString(A4[0] - doc.rightMargin, 25, str(doc.page))
        canvas.restoreState()

    def afterFlowable(self, flowable):
        if (isinstance(flowable, Paragraph) and flowable.style.name == "Section"
                and flowable.getPlainText().split()[0].isdigit()):
            label = flowable.getPlainText()
            key = f"section-{label.split()[0]}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(label, key, level=0, closed=False)
            self.notify("TOCEntry", (0, label, self.page, key))


def styles():
    base = getSampleStyleSheet()
    result = {}
    for name, size, leading in [("Body", 9.7, 14.2), ("Section", 15, 19),
                                ("Subsection", 11.5, 15.5), ("Cell", 8.2, 11.4),
                                ("Code", 7.4, 10), ("TOC", 10, 15)]:
        result[name] = ParagraphStyle(name, fontName="SSD", fontSize=size, leading=leading,
                                      textColor=colors.black, spaceAfter=7,
                                      splitLongWords=True, allowWidows=0, allowOrphans=0)
    for name in ("Section", "Subsection"):
        result[name].fontName = "SSD-Bold"
        result[name].spaceBefore = 16 if name == "Section" else 10
        result[name].keepWithNext = True
    result["Code"].fontName = "SSD-Mono"
    result["Code"].spaceBefore = 4
    result["Code"].spaceAfter = 10
    result["Code"].leftIndent = 6
    result["Cell"].spaceAfter = 0
    result["Title"] = ParagraphStyle("Title", fontName="SSD-Bold", fontSize=28, leading=34,
                                      textColor=colors.black, spaceAfter=24)
    result["Subtitle"] = ParagraphStyle("Subtitle", fontName="SSD", fontSize=13, leading=19,
                                         textColor=colors.black, spaceAfter=14)
    result["Small"] = ParagraphStyle("Small", parent=result["Body"], fontSize=9, leading=13)
    return result


def make_table(lines, st, width):
    rows = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in lines]
    rows = [row for row in rows if not all(re.fullmatch(r":?-+:?", c) for c in row)]
    count = len(rows[0])
    if any(len(row) != count for row in rows):
        raise ValueError(f"Table column mismatch: {rows[:2]}")
    # Short identifiers/status columns stay compact; narrative columns get room.
    scores = []
    for col in range(count):
        lengths = [len(row[col]) for row in rows]
        score = max(7, sum(lengths) / len(lengths)) ** .64
        if rows[0][col] in {"ID", "Estado", "Prioridad", "HTTP", "Código"}:
            score = 3.1 if rows[0][col] != "Estado" else 4.3
        scores.append(score)
    widths = [width * score / sum(scores) for score in scores]
    if count == 3 and rows[0][1] == "HTTP":
        widths = [126, 44, width - 170]
    minimum = 44
    for _ in range(3):
        missing = sum(max(0, minimum - v) for v in widths)
        if missing == 0:
            break
        flexible = sum(v for v in widths if v > minimum)
        widths = [minimum if v < minimum else v - missing * v / flexible for v in widths]
    cells = [[Paragraph(inline(cell), st["Cell"]) for cell in row] for row in rows]
    table = LongTable(cells, colWidths=widths, repeatRows=1, hAlign="LEFT", splitByRow=1)
    commands = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce4ed")),
                ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#d9d9d9")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]
    for row in range(2, len(rows), 2):
        commands.append(("BACKGROUND", (0, row), (-1, row), colors.HexColor("#f5f7fa")))
    table.setStyle(TableStyle(commands))
    return table


def body_flowables(text, st, width):
    lines = text.splitlines()
    flows = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        if line.startswith("# "):
            index += 1
            continue
        if line.startswith("```"):
            index += 1
            block = []
            while index < len(lines) and not lines[index].startswith("```"):
                parts = textwrap.wrap(lines[index], width=104, replace_whitespace=False,
                                      drop_whitespace=False, break_long_words=True,
                                      break_on_hyphens=False) or [""]
                block.extend(parts)
                index += 1
            code = Preformatted("\n".join(block), st["Code"])
            if len(block) <= 40:
                group = [code]
                if (flows and isinstance(flows[-1], Paragraph)
                        and len(flows[-1].getPlainText()) < 120
                        and flows[-1].getPlainText().endswith(":")):
                    group.insert(0, flows.pop())
                while (flows and isinstance(flows[-1], Paragraph)
                       and flows[-1].style.name in {"Section", "Subsection"}):
                    group.insert(0, flows.pop())
                flows.append(KeepTogether(group))
            else:
                flows.append(code)
            index += 1
            continue
        if line.startswith("|"):
            block = []
            while index < len(lines) and lines[index].startswith("|"):
                block.append(lines[index])
                index += 1
            flows.append(make_table(block, st, width))
            flows.append(Spacer(1, 9))
            continue
        if line.startswith("## "):
            flows.append(Paragraph(inline(line[3:]), st["Section"]))
            index += 1
            continue
        if line.startswith("### "):
            flows.append(Paragraph(inline(line[4:]), st["Subsection"]))
            index += 1
            continue
        bullet = line.startswith("- ")
        paragraph = [line[2:] if bullet else line]
        index += 1
        while index < len(lines) and lines[index].strip() and not lines[index].startswith(("#", "|", "```", "- ")):
            paragraph.append(lines[index])
            index += 1
        flows.append(Paragraph(inline(" ".join(paragraph)), st["Body"], bulletText="•" if bullet else None))
    return flows


def build():
    font_setup()
    DEST.parent.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    st = styles()
    document = SSDDocument(DEST)
    source = SOURCE.read_text(encoding="utf-8")
    # The source intro is represented on the cover; all numbered sections follow.
    numbered = source[source.index("## 1 Propósito"):]
    story = [Spacer(1, 56), Paragraph("Especificación del diseño de software de MangaColor", st["Title"]),
             Paragraph("Reescritura de manga colorization v2", st["Subtitle"]),
             Paragraph("Versión 1.0 · 16 de noviembre de 2026", st["Subtitle"]),
             Spacer(1, 22),
             Paragraph("Arquitectura, modelo, control de color, reconstrucción a alta resolución, datos y entrenamiento", st["Body"]),
             Paragraph("Este documento define el comportamiento implementado y los requisitos del producto objetivo. Las capacidades pendientes y los criterios propuestos se identifican de forma explícita.", st["Body"]),
             Spacer(1, 20),
             Paragraph("Base de código 0.1.0 · Documento editable docs/SSD.md", st["Small"]), PageBreak(),
             Paragraph("Contenido", st["Section"])]
    toc = TableOfContents()
    toc.levelStyles = [st["TOC"]]
    story.extend([toc, PageBreak()])
    story.extend(body_flowables(numbered, st, document.width))
    document.multiBuild(story)
    print(json.dumps({"pdf": str(DEST), "markdown_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                      "pdf_sha256": hashlib.sha256(DEST.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    build()
