# Label Studio Manual Labeling Guide (WSL/local)

Setup/run/export steps for manually relabeling `datasets/recycling_
material_real_selected/`'s 91 images -- see `docs/real_selected_manual_
relabeling_plan.md` for why this is happening and the labeling criteria.
This doc is just the tool mechanics.

## 1. Install (dedicated venv)

Keep Label Studio in its own venv -- it's a labeling tool, not a
training dependency, and doesn't need to share `.venv-autolabel`'s
torch/CUDA setup:

```bash
cd ~/Projects/physical-ai-recycling-cell
python3 -m venv .venv-labelstudio
source .venv-labelstudio/bin/activate
pip install --upgrade pip
pip install label-studio
```

## 2. Run

```bash
source .venv-labelstudio/bin/activate
label-studio start
```

This starts a local server and should auto-open a browser tab. If it
doesn't, or you're running headless, open:

```
http://localhost:8080
```

First run asks you to create a local account (email + password -- this
stays on your machine, nothing external). Leave the server running in
this terminal (or a tmux session, same pattern as training runs -- see
`docs/material_v1_augmentation_experiment_plan.md`'s tmux section if you
want it to survive a terminal close) while you label.

## 3. Create the project

1. **Create Project** -> name it e.g. `recycling_real_selected_relabel`.
2. **Data Import**: import images now, or add them after project
   creation (see step 4).
3. **Labeling Setup** -> choose **Object Detection with Bounding
   Boxes**.
4. Replace the generated label config with this exact XML (class order
   matters -- see `docs/real_selected_manual_relabeling_plan.md` section
   8 on why):

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="plastic" background="#2ecc71"/>
    <Label value="metal" background="#3498db"/>
    <Label value="glass" background="#e67e22"/>
    <Label value="paper" background="#e74c3c"/>
  </RectangleLabels>
</View>
```

This defines `plastic=0, metal=1, glass=2, paper=3` in the exported YOLO
`classes.txt`, matching this project's taxonomy.

## 4. Import images

Import all 91 images from `datasets/recycling_material_real_selected/`
(all 4 class subfolders -- `plastic/`, `metal/`, `glass/`, `paper/`).
Two ways to do this:

- **Local Storage import** (recommended for a local-only setup): in the
  project's **Settings -> Cloud Storage -> Add Source Storage**, add a
  **Local files** source pointed at the absolute path
  `/home/rlack/Projects/physical-ai-recycling-cell/datasets/recycling_
  material_real_selected`, then **Sync**. Label Studio needs
  `LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true` and `LABEL_STUDIO_
  LOCAL_FILES_DOCUMENT_ROOT` set to allow this -- set them before
  `label-studio start`:
  ```bash
  export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
  export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/home/rlack/Projects/physical-ai-recycling-cell/datasets
  label-studio start
  ```
- **Direct upload**: drag-and-drop the image files into the **Data
  Import** screen. Simpler for a one-time 91-image batch, but Label
  Studio copies the files into its own storage rather than referencing
  them in place.

Either way works for this task's scale -- local storage import avoids
duplicating 91 images if you prefer.

## 5. Labeling

- Work through the queue one image at a time. For each image: draw one
  box around the target object, assign its class, then **Submit**.
- Follow `docs/real_selected_manual_relabeling_plan.md` sections 4-5 for
  which class to pick and how tightly to draw the box.
- If an image is too ambiguous to label confidently, **Skip** it in
  Label Studio (don't force a box) and record it as `DROP` in `results/
  real_selected_manual_relabeling_tracker.csv` afterward.
- No need to label in the same session -- Label Studio saves per-image
  progress as you go; the project's task list shows what's done vs.
  pending if you stop and come back.

## 6. Export to YOLO format

Project page -> **Export** -> format **YOLO**. This downloads a `.zip`
containing:
- `images/` (copies of the labeled images)
- `labels/` (`.txt` per image, YOLO format: `class_id x_center y_center
  width height`, normalized)
- `classes.txt` (label name per `class_id`, in project label-config
  order -- verify it reads `plastic`/`metal`/`glass`/`paper` on lines
  0-3 before trusting the export)

## 7. Where to put the export

Unzip into a dedicated, gitignored location -- do not put raw exported
labels under `datasets/` casually without checking `.gitignore` first
(the broad `datasets/` ignore rule already covers this, but keep it
namespaced so it's easy to find):

```bash
mkdir -p datasets/recycling_material_real_selected_manual_labels
unzip ~/Downloads/project-*-at-*-yolo.zip \
  -d datasets/recycling_material_real_selected_manual_labels
```

Then, per image, update `results/real_selected_manual_relabeling_
tracker.csv`'s `manual_label_path` column to point at the corresponding
`.txt` file under that directory, and set `labeling_status=LABELED`.
This tracker -- not Label Studio's own project state -- is this
project's source of truth for "which images are ready to use" (see
`docs/real_selected_manual_relabeling_plan.md` section 9).

## Notes

- Label Studio and its exported labels/images are local-only and
  gitignored (`datasets/` is fully ignored -- see `.gitignore`). Only
  the tracker CSV/MD and the docs in this project are meant to be
  committed.
- If you want to stop and resume later, just leave the project as-is --
  Label Studio's own project/task state persists in its local SQLite DB
  (`~/.local/share/label-studio/` by default) between `label-studio
  start` runs.
