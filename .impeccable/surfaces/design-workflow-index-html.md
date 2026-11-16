---
version: 1
slug: "design-workflow-index-html"
primary_target: "design/workflow/index.html"
related_targets: []
---

# Klinik kiritish

Mode: operate. Audience: a cardiologist at a workstation, assembling one case. Task: move through Holat, Bemor, Laboratoriya, EKG, Echo, and Tekshiruv, then open a separate results screen. The prototype does not call the agent.

Memorable moment: the right column rewrites the case in clinical lines as fields change.

Unresolved: when this prototype is wired into Streamlit.

## Direction contract

THESIS: One case is built on a spine — steps at the left, only the current entry in the center, the case rewriting itself on the right — and the conclusion lives on a later screen. The refused default is the single viewport that stacks patient, ECG, and conclusion together.

OWN-WORLD: The clinic already running in the app: ground #ECFEFF, ink #164E63, accent #0891B2, warning #FEF3C7 on #451A03, white work surfaces, Atkinson Hyperlegible. Status is a word (tayyor, kiritilmoqda, hali yo‘q, ogohlantirish). Signature move: the right column rewrites the case in clinical lines as a field changes.

STORY: The cardiologist loads one case, completes patient, lab, ECG, and echo, sees every gap in words, then opens a separate results screen. This prototype does not run the agent and does not state a diagnosis.

FIRST VIEWPORT: A full-height three-column shell. Left, about 280px: “Kardiologik AI agent” and six step buttons; Holat is tayyor, Bemor is the cyan current row. Center: amber warning, the heading Bemor, age and sex in two fields, complaint and history as tall fields, and “Keyingi · Laboratoriya” at the form’s foot. Right, about 340px: the heading Yig‘ilgan holat, facts for source, age and sex, complaint, history, and medicines, then one line naming the next gap. No eyebrow above either heading.

FORM: Chap holat ustuni, first on the ordered list of structures. seed:6aaedc81b7fe81d0

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
