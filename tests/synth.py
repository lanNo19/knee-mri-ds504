"""Synthetic competition data for tests: same layout as Kaggle, made-up content."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FINDINGS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]

REPORTS = {
    "ENGLISH": "The anterior cruciate ligament is intact and the posterior cruciate ligament looks normal. There is a small joint effusion without any loose bodies. No fracture is seen in the distal femur or proximal tibia, and the menisci appear well preserved throughout.",
    "GERMAN": "Das vordere Kreuzband ist intakt. Es zeigt sich ein kleiner Gelenkerguss. Keine Fraktur im distalen Femur oder in der proximalen Tibia.",
    "FRENCH": "Le ligament croisé antérieur est intact. Petit épanchement articulaire. Pas de fracture du fémur distal ni du tibia proximal.",
    "DUTCH": "De voorste kruisband is intact. Er is een kleine hoeveelheid gewrichtsvocht. Geen fractuur van het distale femur of de proximale tibia.",
    "SPANISH": "El ligamento cruzado anterior está íntegro. Pequeño derrame articular. No se observan fracturas en el fémur distal ni en la tibia proximal.",
    "ITALIAN": "Il legamento crociato anteriore è integro. Modesto versamento articolare. Non fratture del femore distale né della tibia prossimale.",
    "PORTUGUESE": "O ligamento cruzado anterior está íntegro. Pequeno derrame articular. Sem fraturas no fêmur distal ou na tíbia proximal.",
}

# Scanner per language; MODEL_X is shared by GERMAN and DUTCH only, so they merge.
SCANNERS = {
    "ENGLISH": ("SIEMENS", "AVANTO", 1.5),
    "GERMAN": ("SIEMENS", "MODEL_X", 3.0),
    "FRENCH": ("GE MEDICAL SYSTEMS", "SIGNA", 1.5),
    "DUTCH": ("SIEMENS", "MODEL_X", 3.0),
    "SPANISH": ("PHILIPS", "INGENIA", 3.0),
    "ITALIAN": ("GE MEDICAL SYSTEMS", "DISCOVERY", 3.0),
    "PORTUGUESE": ("PHILIPS", "ACHIEVA", 1.5),
}

SAG_IOP = [0, 1, 0, 0, 0, -1]  # canonical sagittal
COR_IOP = [1, 0, 0, 0, 0, -1]
AX_IOP = [1, 0, 0, 0, 1, 0]


def uid(i: int, j: int = 0, k: int = 0) -> str:
    return f"1.2.826.0.1.3680043.8.498.{1000 + i}.{j}.{k}"


def write_dicom(path: Path, img: np.ndarray, ipp, iop, inst: int, scanner, laterality: str | None,
                spacing=(0.5, 0.5)):
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = "MR"
    ds.Manufacturer, ds.ManufacturerModelName, ds.MagneticFieldStrength = scanner[0], scanner[1], scanner[2]
    if laterality:
        ds.Laterality = laterality
    ds.ImagePositionPatient = [float(x) for x in ipp]
    ds.ImageOrientationPatient = [float(x) for x in iop]
    ds.PixelSpacing = [float(spacing[0]), float(spacing[1])]
    ds.InstanceNumber = inst
    ds.Rows, ds.Columns = img.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    ds.PixelData = img.astype(np.uint16).tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(path), enforce_file_format=True)


def make_competition(root: Path, n_per_lang: int = 6, n_slices: int = 12, size: int = 48, seed: int = 0,
                     gold_n: int = 12) -> dict:
    """Writes train.csv, train_series.csv, train_series/ and three label tables under root/tables."""
    rng = np.random.default_rng(seed)
    rows, ser, i = [], [], 0
    langs = list(REPORTS)
    for lang in langs:
        for _ in range(n_per_lang):
            st = uid(i)
            side = "R" if i % 2 == 0 else "L"
            cx = -80.0 if side == "R" else 80.0
            tag = side if i % 3 else None  # some studies without the tag
            labels = {f: np.nan for f in FINDINGS}
            rows.append({"StudyInstanceUID": st, "Report": REPORTS[lang] + f" Exam {i}.", **labels})
            for p_i, (plane, iop) in enumerate((("Sagittal", SAG_IOP), ("Coronal", COR_IOP), ("Axial", AX_IOP))):
                for s_i, fs in enumerate((1, 0)):
                    se = uid(i, p_i + 1, s_i)
                    ser.append({"StudyInstanceUID": st, "SeriesInstanceUID": se, "Fluid_Sensitive": fs,
                                "Fat_Suppression": fs, "Anatomical_Plane": plane})
                    for k in range(n_slices):
                        img = (rng.random((size, size)) * 500 + 100 * k).astype(np.uint16)
                        if plane == "Sagittal":
                            ipp = [cx - 30 + 5 * k, -size / 4, size / 4]
                        elif plane == "Coronal":
                            ipp = [cx - size / 4, -20 + 5 * k, size / 4]
                        else:
                            ipp = [cx - size / 4, -size / 4, -20 + 5 * k]
                        # file names in shuffled order so file order is not slice order
                        fname = f"{(k * 7) % n_slices:04d}.dcm"
                        write_dicom(root / "train_series" / st / se / fname, img, ipp, iop, k + 1,
                                    SCANNERS[lang], tag)
            i += 1
    train = pd.DataFrame(rows)
    truth = rng.random((len(train), 12)) < 0.4
    # gold: every (len/gold_n)-th study gets all 12 labels, equal to the truth
    gold_rows = list(range(0, len(train), max(1, len(train) // gold_n)))[:gold_n]
    for j in gold_rows:
        for c, f in enumerate(FINDINGS):
            train.loc[j, f] = int(truth[j, c])
    train.to_csv(root / "train.csv", index=False)
    pd.DataFrame(ser).to_csv(root / "train_series.csv", index=False)
    # label tables
    tdir = root / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    uids = train["StudyInstanceUID"]
    pil = {"StudyInstanceUID": uids}
    for j, f in enumerate(FINDINGS):
        verdict = np.where(truth[:, j], "YES", "NO").astype(object)
        silent = rng.random(len(train)) < 0.2
        verdict[silent] = "UNK"
        # the real table's layout: a derived number, then __conf and __verdict
        pil[f] = np.where(silent, 0, truth[:, j]).astype(int)
        pil[f + "__conf"] = np.where(silent, 0.05, 0.9)
        pil[f + "__verdict"] = verdict
    pd.DataFrame(pil).to_csv(tdir / "report_labels_v2.csv", index=False)
    for name in ("llm_labels_v2.csv", "labels_llm_gpt56sol.csv"):
        noisy = np.clip(truth + rng.normal(0, 0.25, truth.shape), 0, 1)
        d = pd.DataFrame(noisy, columns=FINDINGS)
        d.insert(0, "StudyInstanceUID", uids)
        d.to_csv(tdir / name, index=False)
    return {"n_studies": len(train), "uids": list(uids), "reports": list(train["Report"])}
