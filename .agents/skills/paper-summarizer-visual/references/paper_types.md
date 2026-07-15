# Paper Type Definitions & Summary Structures

This document defines how to classify academic papers and what summary structure
to use for each type. The agent reads the preprocessed markdown and matches
against the classification criteria below.

---

## 🆕 Advanced Section Types (New in v2)
When generating the `summary.json`, you can now use a variety of section types for a richer visual layout. For any section in the `"sections"` array, you can define a `"type"` (defaults to `"card"`).

Supported types and their expected schema:

1. **`card` (Default)**
   ```json
   { "heading": "Title", "icon": "📌", "content": "Markdown text..." }
   ```

2. **`stats`**
   ```json
   { "heading": "Key Metrics", "icon": "📊", "type": "stats", "stats": [
       {"icon": "📈", "value": "95%", "label": "Accuracy"},
       {"icon": "⚡", "value": "1.2s", "label": "Latency"}
     ]
   }
   ```

3. **`timeline`**
   ```json
   { "heading": "Process", "icon": "⏱️", "type": "timeline", "steps": [
       {"title": "Data Prep", "description": "Cleaned the dataset..."},
       {"title": "Training", "description": "Trained the model..."}
     ]
   }
   ```

4. **`table`**
   ```json
   { "heading": "Comparison", "icon": "⚖️", "type": "table",
     "columns": ["Method", "Accuracy", "Speed"],
     "rows": [ ["Baseline", "80%", "Fast"], ["Proposed", "95%", "Slow"] ]
   }
   ```

5. **`quote`**
   ```json
   { "heading": "Core Motivation", "icon": "💬", "type": "quote",
     "quote": "The fundamental problem is scale...",
     "attribution": "Section 2.1"
   }
   ```

6. **`flowchart`**
   ```json
   { "heading": "Architecture", "icon": "🔄", "type": "flowchart", "steps": [
       {"icon": "📥", "label": "Input", "description": "Raw data"},
       {"icon": "⚙️", "label": "Process", "description": "Filtering"},
       {"icon": "📤", "label": "Output", "description": "Results"}
     ]
   }
   ```

7. **`two_column`**
   ```json
   { "heading": "Pros & Cons", "icon": "⚖️", "type": "two_column",
     "left": {"title": "Advantages", "content": "- Fast\n- Cheap"},
     "right": {"title": "Limitations", "content": "- Less accurate\n- High memory"}
   }
   ```

8. **`accordion`**
   ```json
   { "heading": "Deep Dive", "icon": "🔍", "type": "accordion", "items": [
       {"icon": "📐", "title": "Math Details", "content": "Equations..."},
       {"icon": "💻", "title": "Code Snippet", "content": "```python\n...```"}
     ]
   }
   ```

> **IMPORTANT**: Use a mix of these advanced section types to make the summaries highly engaging and visually diverse!

---

## Classification Criteria

Read the first ~300 lines of the preprocessed markdown. Look for these signals:

### 1. `textbook` — Textbook / Practical Guide

**Signals:**
- Chapter/section numbering (Chapter 1, 2, 3…)
- Exercises, examples, or practice problems
- Progressive difficulty structure
- pedagogical tone, Code examples or tool tutorials

**Color theme:** `purple` (gradient: #7C3AED → #A855F7)

**Summary sections:** (Use advanced types like `accordion` for concepts, `stats` for chapter counts, `flowchart` for learning progression)
1. 📖 **Overview** — What the book covers, target audience, prerequisites
2. 📚 **Concepts Catalog** — List of key concepts/topics organized by chapter or theme
3. ⚙️ **Methods & Tools** — Mathematical methods, software tools, libraries introduced
4. 🗺️ **Learning Progression** — How knowledge builds from chapter to chapter
5. 📐 **Key Formulations** — The most important equations, models, or algorithms presented

---

### 2. `research_article` — Research Article

**Signals:**
- Abstract, Introduction, Related Work, Methodology, Results, Conclusion structure
- Novelty claims, Experimental evaluation with datasets
- Peer-reviewed journal/conference format

**Color theme:** `blue` (gradient: #2563EB → #3B82F6)

**Summary sections:** (Use `two_column`, `table`, `stats` for results)
1. 🎯 **Problem Statement** — What problem does the paper address and why it matters
2. 🔬 **Related Work** — Key prior approaches and how this work differs
3. ⚙️ **Methodology** — The proposed approach, model, or algorithm
4. 📊 **Results & Validation** — Key findings, metrics, comparisons
5. 💡 **Contributions** — What's new and what impact it claims

---

### 3. `review_survey` — Review / Survey Paper

**Signals:**
- "survey", "review", "state of the art", "literature review"
- Systematic comparison of multiple approaches, Taxonomy tables

**Color theme:** `green` (gradient: #059669 → #10B981)

**Summary sections:** (Use `timeline` for history, `table` for taxonomies)
1. 🔭 **Scope & Motivation** — What area is surveyed and why now
2. 🗂️ **Taxonomy of Approaches** — How the surveyed methods are categorized
3. ⚖️ **Comparative Analysis** — Strengths/weaknesses of different approaches
4. 📈 **Trends & Gaps** — What's growing, what's missing in the literature
5. 🔮 **Future Directions** — Open problems and recommended research paths

---

### 4. `mathematical` — Mathematical / Methods Paper

**Signals:**
- Heavy formal notation (theorems, proofs), Complexity analysis
- Focus on theoretical results rather than experiments

**Color theme:** `red` (gradient: #DC2626 → #EF4444)

**Summary sections:** (Use `accordion` for proofs, `quote` for theorems)
1. 📌 **Problem Definition** — Formal problem statement in plain language
2. 📝 **Formulation** — Key mathematical formulation
3. 🧮 **Complexity Analysis** — Computational complexity results
4. 🔧 **Algorithm Design** — Core algorithmic idea and approach
5. 🏆 **Theoretical Results** — Main theorems, bounds, guarantees achieved

---

### 5. `case_study` — Case Study / Simulation

**Signals:**
- Specific city, company, or system named, Simulation framework used
- Focus on a real-world application rather than theory

**Color theme:** `orange` (gradient: #EA580C → #F97316)

**Summary sections:** (Use `flowchart` for setup, `stats` for params)
1. 🌍 **Context & Scenario** — What real-world situation is studied
2. 🏗️ **Model Setup** — Architecture of the simulation/model
3. 📋 **Parameters & Data** — Key inputs, datasets, assumptions
4. 📊 **Simulation Results** — Main findings and output analysis
5. 🎯 **Practical Implications** — What practitioners can learn from this

---

### 6. `technical_report` — Technical Report / Policy Document

**Signals:**
- Government agency or organization as publisher
- Recommendations and implementation guidance, large document

**Color theme:** `gold` (gradient: #D97706 → #F59E0B)

**Summary sections:** (Use `two_column`, `quote` for policies)
1. 📋 **Executive Summary** — Top-level findings and recommendations
2. 🏛️ **Background** — Policy context, regulatory landscape, existing standards
3. 📖 **Guidelines & Recommendations** — Core guidance provided
4. 🚀 **Implementation Strategy** — How to deploy/adopt the recommendations
5. 📊 **Impact Assessment** — Expected benefits, costs, success metrics

---

## Fallback

If the paper doesn't clearly match any type, default to `research_article` (blue theme)
and note the uncertainty in the TL;DR field.
