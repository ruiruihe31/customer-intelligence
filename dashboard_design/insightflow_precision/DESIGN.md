# Design System Specification: The Fluid Analytic Intelligence

## 1. Overview & Creative North Star
The North Star for this design system is **"The Digital Curator."** 

In the realm of B2B SaaS, data is often chaotic. Our goal is not just to display information, but to curate it into a high-end editorial experience. We are moving away from the rigid, "boxed-in" feel of traditional dashboards. Instead, we embrace a layout that feels intentional, breathable, and layered. 

By utilizing **intentional asymmetry** and **tonal depth**, we create an environment where the most important insights naturally "float" to the surface. This system prioritizes the "quiet" spaces between data points as much as the data itself, ensuring that even the most complex bubble charts or heatmaps feel sophisticated rather than cluttered.

---

## 2. Colors & Surface Architecture
We move beyond flat hex codes to a system of **Tonal Layering**.

### The "No-Line" Rule
Standard 1px borders are strictly prohibited for sectioning. Boundaries must be defined solely through background color shifts or subtle tonal transitions. 
*   **Example:** A `surface-container-low` data table sitting on a `surface` background provides all the definition needed without the visual "noise" of a stroke.

### Surface Hierarchy & Nesting
Treat the UI as a series of physical layers—like stacked sheets of fine paper.
- **Background (`#f8f9ff`):** The canvas.
- **Surface Container Low (`#eff4ff`):** For large content areas or sidebar backgrounds.
- **Surface Container Lowest (`#ffffff`):** For the primary data cards (KPIs, Charts). This creates a crisp, clean lift against the softer background.
- **Surface Container Highest (`#d3e4fe`):** Reserved for "active" states or nested highlights within a card.

### The "Glass & Gradient" Rule
To achieve a premium, custom feel:
- **Glassmorphism:** Floating elements (like navigation tooltips or modal overlays) should use `surface-container-lowest` at 80% opacity with a `20px` backdrop-blur.
- **Signature Textures:** Use a subtle linear gradient (from `primary` to `primary_container`) for main CTAs to provide a depth that flat `#0fdad7` cannot achieve alone.

---

## 3. Typography: Editorial Authority
We use a dual-font strategy to balance character with readability.

*   **Display & Headlines (Manrope):** Chosen for its geometric precision and modern "tech-forward" personality. Use `headline-lg` (2rem) for dashboard titles to establish an authoritative hierarchy.
*   **Body & Labels (Inter):** The workhorse for data. Inter’s tall x-height ensures that complex data tables and KPI labels remain legible at small scales (`label-sm` 0.6875rem).
*   **Weight as Hierarchy:** Use `Medium` (500) for labels and `SemiBold` (600) for titles. Avoid `Bold` (700) unless it is for extreme emphasis within a `display-sm` context.

---

## 4. Elevation & Depth
Depth is a functional tool, not a decoration.

### The Layering Principle
Achieve hierarchy by "stacking" the surface tiers. A `surface-container-lowest` card placed on a `surface-container-low` section creates a soft, natural lift.

### Ambient Shadows
When a "floating" effect is required (e.g., a dragged card or a dropdown menu), shadows must be extra-diffused.
- **Shadow Token:** `0 12px 32px -4px rgba(11, 28, 48, 0.06)`. 
- **Note:** The shadow color is a tinted version of `on-surface` (`#0b1c30`), ensuring the shadow feels like natural ambient light.

### The "Ghost Border" Fallback
If a border is required for accessibility (e.g., in high-contrast needs), use a **Ghost Border**: `outline-variant` (`#c2c6d9`) at **15% opacity**. 100% opaque, high-contrast borders are forbidden.

---

## 5. Components

### Primary Buttons
- **Style:** Gradient fill (`primary` to `primary_container`), **maximum (pill-shaped)** corner radius.
- **Interaction:** On hover, shift the gradient intensity; on press, use `surface_tint` for a subtle inward glow.

### KPI Cards
- **Structure:** No borders. Use `surface-container-lowest` on top of a `surface` background.
- **Data Visualization:** Micro-sparklines should use `primary` with a 10% opacity fill underneath the line to create a "filled volume" effect.

### Data Tables
- **Styling:** Forbid divider lines between rows. Use alternating row colors (`surface` and `surface-container-low`) or simply **spacious** vertical white space to separate entries.
- **Headers:** Use `label-md` in `on-surface-variant` (`#424656`) with all-caps tracking (+0.05em).

### Input Fields
- **Style:** `surface-container-lowest` with a subtle `outline-variant` Ghost Border. 
- **States:** On focus, the border disappears and is replaced by a 2px `primary` glow with a soft spread.

### Custom Component: The "Insight Ribbon"
A specialized component for this system. A slim, vertical accent bar (4px wide) of `secondary` (`#4b807e`) or `tertiary` (`#ffb347`) placed on the far left of a card to indicate "Success" or "Action Required" without coloring the entire container.

---

## 6. Do’s and Don’ts

### Do:
*   **Do** use asymmetrical margins (e.g., wider left padding than right in a header) to create an editorial, "curated" feel.
*   **Do** use `surface-container` shifts to define the "Sidebar" vs "Main Content" rather than a vertical line.
*   **Do** leverage the **maximum (pill-shaped)** roundedness for large-scale containers to soften the B2B "coldness."

### Don’t:
*   **Don’t** use pure black (`#000000`) for text. Use `on-surface` (`#0b1c30`) to maintain a premium tonal range.
*   **Don’t** use high-intensity red backgrounds for errors. Use `error_container` (`#ffdad6`) with `on_error_container` (`#93000a`) text for a sophisticated, readable alert.
*   **Don’t** crowd the charts. If a chart has more than 5 variables, use a "Pagination" or "Tabbed" pattern to distribute the cognitive load.