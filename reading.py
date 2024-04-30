import io
from typing import Dict

import fitz
from PIL import Image


def recoverpix(doc, item):
    # https://github.com/pymupdf/PyMuPDF-Utilities/blob/master/examples/extract-images/extract-from-pages.py
    xref = item[0]  # xref of PDF image
    smask = item[1]  # xref of its /SMask

    # special case: /SMask or /Mask exists
    if smask > 0:
        pix0 = fitz.Pixmap(doc.extract_image(xref)["image"])
        if pix0.alpha:  # catch irregular situation
            pix0 = fitz.Pixmap(pix0, 0)  # remove alpha channel
        mask = fitz.Pixmap(doc.extract_image(smask)["image"])

        try:
            pix = fitz.Pixmap(pix0, mask)
        except Exception as e:  # fallback to original base image in case of problems
            print(e)
            pix = fitz.Pixmap(doc.extract_image(xref)["image"])

        if pix0.n > 3:
            ext = "pam"
        else:
            ext = "png"

        return {  # create dictionary expected by caller
            "ext": ext,
            "colorspace": pix.colorspace.n,
            "image": pix.tobytes(ext),
        }

    # special case: /ColorSpace definition exists
    # to be sure, we convert these cases to RGB PNG images
    if "/ColorSpace" in doc.xref_object(xref, compressed=True):
        pix = fitz.Pixmap(doc, xref)
        pix = fitz.Pixmap(fitz.csRGB, pix)
        return {  # create dictionary expected by caller
            "ext": "png",
            "colorspace": 3,
            "image": pix.tobytes("png"),
        }
    return doc.extract_image(xref)


def get_doc_images(doc: fitz.Document, min_size: int = 100) -> Dict[int, Image.Image]:
    outputs = dict()
    seen = set()
    for i, page in enumerate(doc.pages()):
        for img in page.get_images(full=True):
            xref = img[0]
            if xref in seen:
                continue

            seen.add(xref)
            width = img[2]
            height = img[3]
            if min(width, height) <= min_size:
                continue
            raw = recoverpix(doc, img)
            imgdata = raw["image"]
            if raw["ext"] == "pam":
                continue

            image = Image.open(io.BytesIO(imgdata), formats=[raw["ext"]])
            outputs.setdefault(i, []).append(image)

    return outputs
