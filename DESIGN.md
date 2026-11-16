---
name: Kardiologik AI agent
description: A flat, one-accent work surface for assembling a single case.
colors:
  ground: "#ECFEFF"
  ink: "#164E63"
  white: "#FFFFFF"
  line: "#A5F3FC"
  wash: "#F0FDFA"
  accent: "#0891B2"
  accent-deep: "#0E7490"
  accent-hover: "#06B6D4"
  on-accent: "#000"
  warning-bg: "#FEF3C7"
  warning-ink: "#451A03"
typography:
  display:
    fontFamily: "Atkinson Hyperlegible, Segoe UI, sans-serif"
    fontSize: "40px"
    fontWeight: 700
    lineHeight: 1.15
  headline:
    fontFamily: "Atkinson Hyperlegible, Segoe UI, sans-serif"
    fontSize: "28px"
    fontWeight: 700
    lineHeight: 1.2
  title:
    fontFamily: "Atkinson Hyperlegible, Segoe UI, sans-serif"
    fontSize: "18px"
    fontWeight: 700
    lineHeight: 1.3
  body:
    fontFamily: "Atkinson Hyperlegible, Segoe UI, sans-serif"
    fontSize: "16px"
    fontWeight: 400
    lineHeight: 1.5
  label:
    fontFamily: "Atkinson Hyperlegible, Segoe UI, sans-serif"
    fontSize: "16px"
    fontWeight: 700
    lineHeight: 1.5
rounded:
  sm: "8px"
  md: "12px"
spacing:
  tight: "6px"
  sm: "8px"
  md: "12px"
  lg: "16px"
  xl: "28px"
components:
  button-primary:
    backgroundColor: "{colors.accent}"
    textColor: "{colors.on-accent}"
    rounded: "{rounded.sm}"
    padding: "0 18px"
    height: "48px"
  button-primary-hover:
    backgroundColor: "{colors.accent-hover}"
    textColor: "{colors.on-accent}"
  button-secondary:
    backgroundColor: "{colors.white}"
    textColor: "{colors.ink}"
    rounded: "{rounded.sm}"
    padding: "0 18px"
    height: "48px"
  button-secondary-hover:
    backgroundColor: "{colors.wash}"
    textColor: "{colors.ink}"
  field:
    backgroundColor: "{colors.white}"
    textColor: "{colors.ink}"
    rounded: "{rounded.md}"
    padding: "12px 14px"
  step:
    backgroundColor: "{colors.white}"
    textColor: "{colors.ink}"
    typography: "{typography.title}"
    rounded: "{rounded.md}"
    padding: "12px"
  step-current:
    backgroundColor: "{colors.accent}"
    textColor: "{colors.on-accent}"
    typography: "{typography.title}"
    rounded: "{rounded.md}"
    padding: "12px"
  alert:
    backgroundColor: "{colors.warning-bg}"
    textColor: "{colors.warning-ink}"
    typography: "{typography.body}"
    rounded: "{rounded.md}"
    padding: "12px 14px"
  choice:
    backgroundColor: "{colors.white}"
    textColor: "{colors.ink}"
    typography: "{typography.body}"
    rounded: "{rounded.md}"
    padding: "12px 14px"
  fact:
    textColor: "{colors.ink}"
    typography: "{typography.body}"
    padding: "10px 0"
---

# Design System: Kardiologik AI agent

## Overview

**Creative North Star: "The Case Spine"**

The Case Spine is a full-height work surface for one case. Steps sit in a narrow column on the left, the current entry fills the center, and the case rewrites itself in the column on the right. The conclusion is a later screen. This is an operate surface: dense, quiet, and meant to be read at a workstation. The screen assembles the case and leaves the decision with the physician.

The palette is a pale cyan ground, white columns, and a single cyan accent. Caution is a full amber band. Status is written as a word. Secondary lines use the deep accent. Surfaces are flat, and depth is a change of tone: ground, white column, amber band.

Motion is rare. When a fact changes, that row washes for 700ms and settles. If the reader prefers reduced motion, the wash is removed. Type is one face, Atkinson Hyperlegible, in two weights: 400 for reading and 700 for headings, steps, actions, and labels.

**Key Characteristics:**

- One cyan accent on a pale ground and white columns
- Status carried by a word, with the current step in black on the accent
- A full amber warning band
- Tabular numerals on clinical figures
- Flat surfaces, with depth made by tone
- Three independently scrolling columns, and a sticky step rail when the window is narrow

## Colors

A pale clinic ground, white work columns, one cyan accent, and an amber band for caution. The system is light.

### Primary

- **Signal Cyan** (#0891B2): the current step, the primary action, the focus ring, the text caret, and the radio. It is the only accent.
- **Deep Signal** (#0E7490): the border of the current step and the primary action, and every secondary line (step status, field hints, fact labels).
- **Lifted Cyan** (#06B6D4): the primary action’s hover fill. Same accent, used only as that hover.

### Neutral

- **Clinic Ground** (#ECFEFF): the page ground, which shows through the center column, and the note under the case facts.
- **Deep Teal Ink** (#164E63): body text, field text, resting steps, and the secondary action.
- **Work White** (#FFFFFF): the side columns, fields, resting steps, secondary actions, and tables.
- **Selection Line** (#A5F3FC): 1px hairlines, field borders, text selection, and the first frame of the fact wash. Selected text stays in the ink color.
- **Hover Wash** (#F0FDFA): the secondary action’s hover fill.
- **On-Accent Black** (#000): type on Signal Cyan. The current step, including its status word, and the primary action use black on the accent (about 5.7:1).

### Warning

- **Amber Band** (#FEF3C7): the full warning surface across the entry, and the field-error surface.
- **Amber Ink** (#451A03): text on the amber band and on a field error.

### Named Rules

**The Status-in-Words Rule.** Status is a word — tayyor, hali yo‘q, ogohlantirish, kutilmoqda, or kiritilgan. Color does not carry status by itself. On the current step the word is black on the accent.

**The Single Accent Rule.** Signal cyan is the only accent. The amber pair is the warning surface. Secondary text is Deep Signal.

## Typography

**Display Font:** Atkinson Hyperlegible (with Segoe UI, sans-serif)
**Body Font:** Atkinson Hyperlegible (with Segoe UI, sans-serif)

**Character:** One face, loaded at 400 and 700. Reading stays at 400. Anything that names a step, a field, or an action is 700. The voice is plain and clinical, set for a long shift at a workstation.

### Hierarchy

- **Display** (700, 40px, 1.15): the entry heading. Below the compact breakpoint this heading is 32px.
- **Headline** (700, 28px, 1.2): the column heading on the case side.
- **Title** (700, 18px, 1.3): step names and the name in the rail. Actions use the same size and weight at a 1.2 line-height.
- **Body** (400, 16px, 1.5): running text, held near 68 characters. The status word under a step is this size, in Deep Signal.
- **Label** (700, 16px, 1.5): field names, in sentence case, with 6px between the name and the control. A field value is 20px, weight 400, line-height 1.4.

### Named Rules

**The Tabular Numerals Rule.** Lab values, rates, and table cells use tabular numerals so columns of figures stay aligned.

## Layout

The shell is three full-height columns: a 280px step rail, a flexible center, and a 340px case panel. Each column scrolls on its own. The center is the ground; the rail and the case panel are white, separated by a 1px selection-line rule. Rail padding is 22px 16px 28px. The center is 22px 28px 36px. The case panel is 28px 18px 32px.

Stack rhythm is 6px between steps, 12px between fields, and 16px under a heading. Two related fields sit side by side with a 12px gap. The actions at the foot of an entry sit apart, with 12px between them.

At 960px and below, the shell stacks in source order. The step rail sticks to the top and becomes a horizontal scroller (8px gap, each step at least 148px wide). Fields fall to one column. Entry actions become full-width and stack. The case panel follows the entry. The center’s side padding becomes 16px.

### Named Rules

**The Own Scroll Rule.** The three columns fill the viewport and each one scrolls on its own. At 960px and below, the step rail sticks to the top, fields stack in one column, and the case panel follows the entry.

## Elevation & Depth

There is no shadow. Depth is tonal. The page sits on Clinic Ground, the rail and the case panel sit on Work White, and caution sits on the amber band. A changed fact washes in place.

### Named Rules

**The Tonal Depth Rule.** Surfaces are flat. Depth is Clinic Ground, then Work White, then the amber band.

## Shapes

Corners are gently curved and there are two sizes. Work pieces — fields, steps, the warning band, the field error, tables, choice rows, and the note under the facts — use a 12px corner. Actions, including the file control, use an 8px corner.

Borders are a 1px selection-line stroke. The current step and the primary action use a 1px deep-accent stroke. A resting step keeps a transparent 1px stroke, so the accent border it gains on hover does not shift the label. Tables clip their cells to the 12px corner.

### Named Rules

**The Two Radii Rule.** Fields, steps, the warning band, tables, and the case note use a 12px corner. Actions use an 8px corner.

## Components

Controls are flat and quiet. Signal Cyan is reserved for the primary action and the current step. Keyboard focus is always a 3px Signal Cyan ring, offset 2px.

### Buttons

- **Shape:** an 8px corner, 48px tall, padding 0 18px. Type is 18px, weight 700, line-height 1.2.
- **Primary:** Signal Cyan fill, On-Accent Black type, 1px Deep Signal border. Hover swaps the fill for Lifted Cyan and leaves the type and the border.
- **Hover / Focus:** the hover above; focus is the 3px accent ring, 2px outside the control.
- **Secondary:** Work White fill, Deep Teal Ink type, 1px Selection Line border. Hover turns the border Signal Cyan and the fill Hover Wash.

### Cards / Containers

- **Corner Style:** 12px on the case note, the table, and the empty work box.
- **Background:** Work White for the columns and the table. The note under the facts returns to Clinic Ground.
- **Shadow Strategy:** none. See Elevation & Depth.
- **Border:** the columns meet on a 1px Selection Line. Table rows are separated by the same 1px line. Fact rows use that line along the top only.
- **Internal Padding:** fact rows 10px 0; the case note and table cells 10px 12px.

### Inputs / Fields

- **Style:** Work White fill, Deep Teal Ink type, 1px Selection Line border, 12px corner, padding 12px 14px. The value is 20px at weight 400 and line-height 1.4. The caret is Signal Cyan.
- **Focus:** the 3px accent ring, offset 2px.
- **Error:** the message under the field reuses Amber Band and Amber Ink, with a 12px corner and padding 8px 12px.
- **Numeric:** lab values add tabular numerals.
- **File:** the file button is 44px tall, 8px corner, Work White, 1px Selection Line border, 16px bold.
- **Choice:** a white row, 12px corner, 1px Selection Line border, padding 12px 14px. The radio is 18px, colored with Signal Cyan, in a 22px track.

### Navigation

The step rail is the navigation. Each step is a white block with a 12px corner and 12px padding. The name is 18px bold ink. Under it, the status word is 16px regular Deep Signal. Steps stack with 6px between them. Hover on a resting step draws a Signal Cyan border.

The current step is filled with Signal Cyan. Its name and its status word are On-Accent Black, and its border is Deep Signal.

At 960px and below, the rail is a sticky horizontal scroller and each step sizes to its label.

### Warning band

A full amber band at the top of the entry: Amber Ink on Amber Band, 12px corner, padding 12px 14px, 16px type. It spans the entry column. A field error is the same pairing at a tighter padding.

### Fact row

The case panel is a list of facts. The label is 16px bold Deep Signal; the value is body ink. When a value changes, the row washes from Selection Line to transparent in 700ms on cubic-bezier(0.16, 1, 0.3, 1). Under prefers-reduced-motion, the wash is removed and the new value simply appears.

## Do's and Don'ts

### Do:

- **Do** write status as one of these words: tayyor, hali yo‘q, ogohlantirish, kutilmoqda, kiritilgan.
- **Do** put black text on the accent for the current step and the primary action.
- **Do** paint caution as a full amber band, amber ink on Amber Band, with a 12px corner.
- **Do** set secondary text in the deep accent.
- **Do** draw a 3px accent focus ring, offset 2px, on keyboard focus.
- **Do** use tabular numerals on clinical figures.
- **Do** wash a changed fact row for 700ms on cubic-bezier(0.16, 1, 0.3, 1), and disable that animation under prefers-reduced-motion.

### Don't:

- **Don't** add a second accent, gray secondary text, or a drop shadow.
- **Don't** communicate status by color alone.
- **Don't** place an eyebrow or kicker above a heading, and don't use emoji as icons.
- **Don't** set headings or field labels in uppercase.
- **Don't** invent a font weight between 400 and 700.
