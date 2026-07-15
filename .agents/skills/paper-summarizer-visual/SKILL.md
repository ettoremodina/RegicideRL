---
name: paper-summarizer-visual
description: >-
  Preprocesses academic PDFs into markdown and images, classifies paper type
  (textbook, research article, review, mathematical, case study, report),
  summarizes with a type-tailored section structure, and renders a colorful
  infographic HTML page. Use when a user provides a PDF path and wants a
  visual, structured summary of the paper.
---

# Paper Summarizer Visual

Converts a PDF academic paper into a visually rich, self-contained HTML summary
page. The workflow preprocesses the PDF (extracting text and images), classifies
the paper type, summarizes using a type-specific structure, and renders a
colorful infographic-style HTML page.

## Dependencies

- **`uv` skill** — Required to run the CLI script with inline dependencies.

## Quick Start

When the user provides a PDF path:

```
1. Preprocess the PDF   →  markdown + images
2. Classify paper type  →  one of 6 categories
3. Summarize content    →  structured JSON
4. Render HTML          →  colorful infographic page
```

## Utility Script

**Location:** `scripts/paper_tool.py` (relative to this skill's directory)

### Subcommand: `preprocess`

Converts a PDF to markdown and extracts embedded images.

```bash
uv run <skill-dir>/scripts/paper_tool.py preprocess \
  --pdf <input.pdf> \
  --output-dir <work-dir>
```

**Outputs:**
- `<work-dir>/content.md` — Full markdown text
- `<work-dir>/images/` — Extracted figures (PNG/JPEG, named `fig_001.png`, etc.)
- `<work-dir>/metadata.json` — Page count, word count, detected title/authors

**If this fails** with a short-text warning, ask the user for guidance — the PDF
is likely scanned/image-only.

### Subcommand: `render`

Fills the HTML template with a structured JSON summary to produce a
self-contained HTML page.

```bash
uv run <skill-dir>/scripts/paper_tool.py render \
  --summary <work-dir>/summary.json \
  --images-dir <work-dir>/images \
  --template <skill-dir>/resources/template.html \
  --output <output-dir>/<paper_name>.html
```

## Workflow

### Step 1: Preprocess the PDF

Run the `preprocess` subcommand on the user's PDF. This creates a working
directory with the extracted markdown and images.

Choose a working directory name based on the PDF filename, e.g.:
- PDF: `references/06_Optimization_Problems.pdf`
- Work dir: `summarized_papers/_work_06/`

### Step 2: Classify the Paper Type

Read the file `references/paper_types.md` (in this skill's directory) to load
the classification criteria.

Then read the first ~300 lines of `<work-dir>/content.md` and classify the
paper into one of these types:

| Type ID | Label |
|---|---|
| `textbook` | Textbook / Practical Guide |
| `research_article` | Research Article |
| `review_survey` | Review / Survey |
| `mathematical` | Mathematical / Methods |
| `case_study` | Case Study / Simulation |
| `technical_report` | Technical Report / Policy |

Use the classification signals described in `references/paper_types.md`.
If uncertain, default to `research_article`.

### Step 3: Summarize the Paper

Read the full `<work-dir>/content.md`. If it is very long (>800 lines), read it
in chunks and build the summary incrementally.

Based on the classified paper type, use the **corresponding section structure**
from `references/paper_types.md`. Each paper type has 5 specific sections with
prescribed headings and icons.

Write the summary as a JSON file at `<work-dir>/summary.json` following this
exact schema:

```json
{
  "title": "Full Paper Title",
  "authors": "Author1, Author2, ...",
  "year": "2024",
  "paper_type": "research_article",
  "color_theme": "blue",
  "tags": ["keyword1", "keyword2", "keyword3"],
  "tldr": "One-sentence summary of the paper's core contribution.",
  "sections": [
    {
      "heading": "Section Heading (from paper_types.md)",
      "icon": "🎯",
      "content": "Markdown-formatted content for this section. Can include **bold**, *italic*, lists, and $math$."
    }
  ],
  "key_figures": ["fig_001.png", "fig_003.png"],
  "key_takeaways": [
    "First key takeaway",
    "Second key takeaway",
    "Third key takeaway"
  ]
}
```

**Color theme mapping:**

| Paper Type | color_theme |
|---|---|
| `textbook` | `purple` |
| `research_article` | `blue` |
| `review_survey` | `green` |
| `mathematical` | `red` |
| `case_study` | `orange` |
| `technical_report` | `gold` |

**Guidelines for section content:**
- Write in clear, concise markdown
- Use bullet points for lists of methods, tools, or findings
- Include key equations using LaTeX math notation (`$...$` or `$$...$$`)
- Reference specific figures from the images directory if relevant
- Keep each section focused — aim for 100-300 words per section
- For `key_figures`, list only the most informative images (diagrams, charts,
  architecture figures) — skip decorative or low-quality images. Check which
  images were extracted by listing `<work-dir>/images/`.

### Step 4: Render the HTML

Run the `render` subcommand to produce the final HTML page.

The output HTML should go to the user's desired output directory. A sensible
default is `summarized_papers/<paper_name>.html` alongside existing markdown
summaries.

### Step 5: Report to User

Tell the user:
1. What paper type was detected
2. Where the HTML file was saved
3. That they can open it in any browser

## Common Mistakes

1. **Forgetting to read `paper_types.md` before classifying.** The classification
   criteria and section structures are in that file — always read it first.

2. **Trying to read the entire content.md at once for very large papers.** If the
   file is >800 lines, read it in chunks (e.g., 500 lines at a time) and build
   the summary incrementally.

3. **Including all extracted images in `key_figures`.** Many extracted images are
   logos, headers, or low-quality fragments. Only include figures that are
   genuinely informative (diagrams, charts, architecture figures, results plots).
