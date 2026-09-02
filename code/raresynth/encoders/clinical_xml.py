"""
Parse TCGA's BCR clinical XML format.

Confirmed real structure (not assumed) from a live downloaded LUAD file:

  <luad:tcga_bcr xmlns:luad="...luad/2.7" xmlns:clin_shared="...shared/2.7"
                 xmlns:shared="...shared/2.7" ...>
    <admin:admin> ... </admin:admin>
    <luad:patient>
      <clin_shared:vital_status ...>Alive</clin_shared:vital_status>
      <clin_shared:age_at_initial_pathologic_diagnosis ...>75</...>
      <shared:gender ...>FEMALE</shared:gender>
      <clin_shared:days_to_last_followup ...>106</...>
      ...
      <!-- a follow_up sub-record repeats several of the same field names -->
      <clin_shared:vital_status ...>Alive</clin_shared:vital_status>
      <clin_shared:days_to_last_followup ...>515</...>
    </luad:patient>
  </luad:tcga_bcr>

Two things a naive parser gets wrong here, both confirmed from the real
file, not guessed:

  1. The root element's namespace prefix (and every child's XSD version) is
     PROJECT-SPECIFIC -- "luad:" here, "brca:" for breast cancer, etc, each
     with its own XSD schema. A parser that matches on the exact namespace
     URI or prefix would need one branch per cancer type. Matching on the
     LOCAL tag name only (stripping the namespace) works across essentially
     any TCGA project without per-project code.

  2. Several fields appear MORE THAN ONCE in a single file -- a follow_up
     sub-record repeats vital_status, days_to_last_followup, etc. from the
     main patient record. Taking the first occurrence blindly can silently
     report stale data (e.g. an early "Alive" from before a later follow-up
     recorded death). Aggregation policy here, chosen deliberately:
       - vital_status: "Dead" if ANY occurrence says Dead, else "Alive" if
         any says Alive, else unknown -- worst-case/most-informative wins.
       - days_to_death / days_to_last_followup: MAX across occurrences --
         the longest known follow-up is the most information we have.
       - everything else (age, gender, site, histology, stage): first
         non-empty occurrence, since these should not meaningfully change
         within one patient's record.

Not every field exists in every cancer type's schema -- confirmed directly
(this LUAD file has no neoplasm_histologic_grade at all, a field other
cancer types do use). All extraction here is optional/defensive: a missing
field is omitted from the output dict, never guessed or defaulted to a
misleading placeholder.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

# fields to extract, and how to aggregate if they appear more than once.
# "first" = first non-empty value; "max_numeric" = largest numeric value;
# "worst_status" = special-cased vital_status logic (see local_name docs)
FIELD_AGGREGATION = {
    "gender": "first",
    "age_at_initial_pathologic_diagnosis": "first",
    "tumor_tissue_site": "first",
    "histological_type": "first",
    "pathologic_stage": "first",
    "neoplasm_histologic_grade": "first",
    "vital_status": "worst_status",
    "days_to_death": "max_numeric",
    "days_to_last_followup": "max_numeric",
}


def _local_name(tag):
    """'{http://...}vital_status' -> 'vital_status'."""
    return tag.split("}", 1)[-1] if "}" in tag else tag


def parse_tcga_clinical_xml(path):
    """Return a flat dict of extracted, aggregated clinical fields.

    Only includes keys that were actually found in the file -- a missing
    field is simply absent from the dict, not set to None/0/"unknown".
    """
    tree = ET.parse(path)
    root = tree.getroot()

    collected = {field: [] for field in FIELD_AGGREGATION}
    for elem in root.iter():
        name = _local_name(elem.tag)
        if name in FIELD_AGGREGATION and elem.text and elem.text.strip():
            collected[name].append(elem.text.strip())

    out = {}
    for field, values in collected.items():
        if not values:
            continue
        policy = FIELD_AGGREGATION[field]
        if policy == "first":
            out[field] = values[0]
        elif policy == "max_numeric":
            numeric = [float(v) for v in values if _is_number(v)]
            if numeric:
                out[field] = max(numeric)
        elif policy == "worst_status":
            lowered = [v.lower() for v in values]
            if any("dead" in v for v in lowered):
                out[field] = "Dead"
            elif any("alive" in v for v in lowered):
                out[field] = "Alive"
            else:
                out[field] = values[0]  # unrecognized value, keep as-is rather than guess
    return out


def _is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def format_tcga_clinical_text(fields, project=None):
    """Turn the extracted field dict into a short clinical summary for
    ClinicalEncoder to embed. Only mentions fields that were actually
    present -- never fabricates a sentence about missing data.
    """
    parts = []

    age = fields.get("age_at_initial_pathologic_diagnosis")
    gender = fields.get("gender")
    if age or gender:
        age_str = f"{int(float(age))}-year-old" if age else "unknown-age"
        gender_str = gender.lower() if gender else "individual"
        parts.append(f"Patient is a {age_str} {gender_str}.")

    site = fields.get("tumor_tissue_site")
    histology = fields.get("histological_type")
    if site or histology:
        bits = [b for b in [site, histology] if b]
        parts.append(f"Diagnosis: {', '.join(bits)}.")

    stage = fields.get("pathologic_stage") or fields.get("neoplasm_histologic_grade")
    if stage:
        parts.append(f"Stage/grade: {stage}.")

    status = fields.get("vital_status")
    days_death = fields.get("days_to_death")
    days_followup = fields.get("days_to_last_followup")
    if status:
        if status == "Dead" and days_death is not None:
            parts.append(f"Vital status: deceased at {int(days_death)} days "
                        f"from diagnosis.")
        elif status == "Alive" and days_followup is not None:
            parts.append(f"Vital status: alive, followed for "
                        f"{int(days_followup)} days.")
        else:
            parts.append(f"Vital status: {status}.")

    if project:
        parts.insert(0, f"Case from the {project} cohort.")

    return " ".join(parts) if parts else "No clinical information recorded."
