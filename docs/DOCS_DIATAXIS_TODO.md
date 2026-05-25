# Documentation restructuring — Diátaxis pass

**Status as of commit `ed422d6`.** The repo has ~25 well-written
top-level `docs/*.md` files plus a strong notebook track, but they
mix the four Diátaxis modes (tutorial / how-to / reference /
explanation) within the same file boundaries. The next consolidation
pass restructures the documentation around user need, not around
internal code structure.

**Priority: this pass comes before any new Phase 2 N-camera BA
work.** A user who lands on the repo today must reach a successful
first calibration without having to read a paper. Until that is true,
algorithm work is on hold (per `## Active priority: make develop
publishable` above).

Delete this file once the acceptance checklist at the bottom is fully
checked.

## The four Diátaxis modes — operating definitions

- **Tutorial.** A guided learning experience. The user is taken from
  zero to a visible result with minimal options and minimal theory.
  Optimised for *success*.
- **How-to guide.** A task recipe. The user already knows what they
  want; the page tells them exactly how to do it. Optimised for
  *getting the job done*.
- **Reference.** The dry, exhaustive description of the machine.
  Optimised for *correctness and lookup*. No teaching, no narrative.
- **Explanation.** The "why" — design choices, alternatives, gauges,
  scientific positioning. Optimised for *understanding*. Not a recipe.

Reference: <https://diataxis.fr>.

## Current state per mode

| Mode | What we have | Diagnostic |
|---|---|---|
| Tutorial | `docs/TUTORIAL.md` (single, too ambitious), notebooks 00–11 | `TUTORIAL.md` jumps from zero to full CMO+BIC in one go — split required |
| How-to | `BRING_YOUR_OWN_DATA`, `FIX_MY_CALIBRATION`, `FROM_OPENCV_TO_STEREOCOMPLEX`, `IDENTIFY_MY_OPTICS`, `NONCENTRAL_FROM_IMAGES`, `REAL_CMO_PYCASO_RAYFIELD` | Good titles, but not surfaced as a how-to index; some pages drift into explanation |
| Reference | `PUBLIC_API.md`, docstrings | Solid base; result-object dataclasses and stability levels need their own pages |
| Explanation | `CMO_PHYSICAL_MODEL`, `CMO_MODEL_SELECTION`, `DIRECT_VS_RAYFIELD_INVERSION`, `PARALLEL_PLATE_ORIGIN_FIELD`, `REAL_CMO_PYCASO_RAYFIELD` | Strong content, but lives next to onboarding docs and overloads first-read pages |

## Target architecture

```
docs/
├── START_HERE.md                       (router — kept as-is, slightly trimmed)
│
├── tutorials/
│   ├── 01_first_calibration.md         (OpenCV path, dataset bundled, success in 10 min)
│   ├── 02_opencv_plus_ray2d.md         (refine the same calibration with Ray2D)
│   ├── 03_first_central_rayfield.md    (move to a 3-D rayfield, triangulate a point)
│   └── 04_first_noncentral_rayfield.md (synthetic non-central example, no CMO yet)
│
├── how_to/
│   ├── INDEX.md                        (problem → guide table)
│   ├── bring_your_own_data.md          (move from docs/)
│   ├── fix_my_calibration.md           (idem)
│   ├── export_back_to_opencv.md        (new — extract from PUBLIC_API)
│   ├── assess_calibration_quality.md   (new — wraps RECONSTRUCTION_API + metrics)
│   ├── fit_noncentral_from_images.md   (move NONCENTRAL_FROM_IMAGES.md)
│   ├── identify_optical_model.md       (move IDENTIFY_MY_OPTICS.md)
│   └── reconstruct_3d_points.md        (move RAYFIELD3D_RECONSTRUCTION.md or merge)
│
├── reference/
│   ├── INDEX.md
│   ├── public_api.md                   (move PUBLIC_API.md — strip narrative, keep tables)
│   ├── result_objects.md               (new — list every public dataclass, fields, units)
│   ├── rayfield_classes.md             (new — ZernikeRayField, central rayfield, etc.)
│   ├── physical_model_classes.md       (new — CMO physical / telecentric / warped, with
│                                        explicit param-vector sizes 19/21, 12/14/16, 26)
│   ├── file_formats.md                 (new — model_io.py JSON schemas)
│   └── stability_levels.md             (move from RELEASE_READINESS.md)
│
└── explanation/
    ├── INDEX.md
    ├── why_rayfields.md                (new — distillation of README + ARCHITECTURE)
    ├── ray2d_vs_3d.md                  (new — 2-D refinement is not a 3-D camera model)
    ├── central_vs_noncentral.md        (new — pinhole assumption, when it fails)
    ├── gauge_choices.md                (new — transverse gauge, fixed f_x, effective
                                          descriptors)
    ├── ray_space_bic.md                (move CMO_MODEL_SELECTION.md, trim recipes)
    ├── cmo_case_study.md               (move CMO_PHYSICAL_MODEL + REAL_CMO_PYCASO_RAYFIELD)
    ├── direct_vs_rayfield.md           (move DIRECT_VS_RAYFIELD_INVERSION.md)
    └── validation_limits.md            (new — what we have NOT validated externally)
```

The on-disk move can stay incremental — what matters is that every
existing file is **classified** and the front pages route correctly.

## Hard rules for this pass

1. **One Diátaxis mode per file.** If a page is currently teaching AND
   doing recipe AND explaining, split it. Don't try to be all three.
2. **The first-read tutorial cannot mention CMO.** A user lands, runs
   a script, sees a number. CMO/Pycaso belongs in the
   explanation/case-study track.
3. **The reference is austere.** No narrative, no motivation, no "why",
   no exemplar paragraph. Tables, signatures, units, shapes, return
   types, stability levels. Read it like an API spec.
4. **Every how-to opens with the user need, not the function name.**
   *"I have left/right folders and a bad OpenCV calibration. What do I
   run?"* — that's the right opening sentence.
5. **Every explanation page declares its kind in the first paragraph.**
   *"This is an explanation page. It does not contain commands you
   need to run."* This avoids users hunting for code in a why-page.
6. **`START_HERE.md` is the only branching router.** All four mode
   indexes link back to it.

## Concrete first commits (small enough to land independently)

These can be done in any order; each is one focused commit.

1. **Tutorial split.** Replace `docs/TUTORIAL.md` with
   `docs/tutorials/01_first_calibration.md` that takes the user from
   downloading a bundled dataset to a single calibration result in
   under 10 minutes, **without** Zernike, BIC, or CMO. Then split the
   rest of the current `TUTORIAL.md` into 02/03/04.
2. **How-to index.** Add `docs/how_to/INDEX.md` with the problem→guide
   table from this file. Move the existing how-to MDs into the
   subdirectory and update the internal links. Keep the old paths as
   stub redirects (one line each) for one release cycle.
3. **Reference: result objects.** Create `docs/reference/result_objects.md`.
   For every public dataclass in `api/`, `metrics/`,
   `benchmarks/`, list every field with type, unit, shape, physical
   meaning. Cross-reference from `PUBLIC_API.md`.
4. **Reference: physical model classes.** Create
   `docs/reference/physical_model_classes.md`. List the four CMO
   families (paraxial, telecentric, warped, plus paper's 26 + SE(3))
   with their parameter-vector layouts and the exact functions that
   produce/consume each. This kills the recurring confusion between
   the 19/21p paraxial and the 26p paper model.
5. **Explanation extract.** Move `CMO_PHYSICAL_MODEL.md`,
   `CMO_MODEL_SELECTION.md`, `DIRECT_VS_RAYFIELD_INVERSION.md`,
   `PARALLEL_PLATE_ORIGIN_FIELD.md`, `REAL_CMO_PYCASO_RAYFIELD.md`
   under `docs/explanation/`. Add a one-paragraph kind-declaration at
   the top of each. Trim any code recipes that belong in a how-to.
6. **Link audit.** Replace every `blob/main/...` Colab/source URL with
   `blob/main/...` *only* if the file actually exists on `main` (post
   `ed422d6`). The temporary `develop`-pointing hack is now obsolete;
   any remaining `blob/main/...` in `docs/` is a bug to fix.
7. **Notebook role.** Update `docs/NOTEBOOKS.md` to say notebooks are
   **case studies** that double as tutorials, and explicitly map each
   notebook to its Diátaxis mode (most are tutorial, a few are
   explanation, the Pycaso ones are case studies).

## Acceptance gates

The Diátaxis pass is **DONE** when:

- The four directories `docs/tutorials/`, `docs/how_to/`,
  `docs/reference/`, `docs/explanation/` exist and each has an
  `INDEX.md`.
- `docs/START_HERE.md` routes to exactly those four indexes (plus
  the notebook track).
- `docs/TUTORIAL.md` either is gone or is a one-line redirect.
- A new user can run the first tutorial end-to-end on a fresh clone
  without reading any other doc, and gets a numerical result.
- The reference directory lists every public dataclass with units and
  shapes.
- No how-to page contains a 5+ line *why* paragraph; no explanation
  page contains a runnable recipe.
- All internal links resolve (run a markdown link checker).

When all gates pass, delete this file and the pointer in `CLAUDE.md`.

## After this pass

Phase 2 N-camera BA (the CDC below in `CLAUDE.md`) can resume.
