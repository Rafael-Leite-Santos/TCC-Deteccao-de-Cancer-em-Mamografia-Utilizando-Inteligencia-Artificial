"""Gera test.dcm: mamografia sintética (16 bits, MONOCHROME2, Modality MG)."""

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

LINHAS, COLUNAS = 2560, 2048
MG_STORAGE = "1.2.840.10008.5.1.4.1.1.1.2"  # Digital Mammography X-Ray Image Storage

rng = np.random.default_rng(42)
yy, xx = np.mgrid[0:LINHAS, 0:COLUNAS].astype(np.float32)

# Mama em formato de semi-elipse encostada na borda esquerda
dist = np.sqrt((xx / (COLUNAS * 0.78)) ** 2 + ((yy - LINHAS / 2) / (LINHAS * 0.45)) ** 2)
mama = np.clip(1.0 - dist, 0, 1) ** 0.35
img = 300 + mama * 2600 + rng.normal(0, 60, (LINHAS, COLUNAS)) * (mama > 0)

# Alguns nódulos claros (cistos simulados)
for cy, cx, r in [(1300, 1050, 110), (800, 600, 70), (1800, 700, 55)]:
    img += 900 * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r**2)))

pixels = np.clip(img, 0, 4095).astype(np.uint16)

meta = FileMetaDataset()
meta.MediaStorageSOPClassUID = MG_STORAGE
meta.MediaStorageSOPInstanceUID = generate_uid()
meta.TransferSyntaxUID = ExplicitVRLittleEndian

ds = FileDataset("test.dcm", {}, file_meta=meta, preamble=b"\0" * 128)
ds.SOPClassUID = MG_STORAGE
ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
ds.StudyInstanceUID = generate_uid()
ds.SeriesInstanceUID = generate_uid()
ds.Modality = "MG"
ds.PatientName = "SINTETICO^TESTE"
ds.PatientID = "000000"
ds.ImageLaterality = "L"
ds.ViewPosition = "CC"
ds.Rows, ds.Columns = LINHAS, COLUNAS
ds.SamplesPerPixel = 1
ds.PhotometricInterpretation = "MONOCHROME2"
ds.BitsAllocated = 16
ds.BitsStored = 12
ds.HighBit = 11
ds.PixelRepresentation = 0
ds.RescaleSlope = 1
ds.RescaleIntercept = 0
ds.WindowCenter = 2000
ds.WindowWidth = 3600
ds.PixelData = pixels.tobytes()
ds.is_little_endian = True
ds.is_implicit_VR = False
ds.save_as("test.dcm", write_like_original=False)
print(f"test.dcm gerado ({COLUNAS}x{LINHAS}, 16 bits)")
