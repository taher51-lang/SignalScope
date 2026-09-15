from PIL import Image
from PIL.ExifTags import TAGS
import io

def extract_metadata(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes))
    exif_data = img._getexif() if hasattr(img, "_getexif") else None

    if not exif_data:
        return {
            "has_exif": False,
            "note": "No EXIF metadata found — common for AI-generated images, but also common for screenshots or edited real photos."
        }

    readable_exif = {}
    for tag_id, value in exif_data.items():
        tag_name = TAGS.get(tag_id, tag_id)
        readable_exif[tag_name] = str(value)

    # A few fields that matter most for authenticity signal
    camera_make = readable_exif.get("Make")
    camera_model = readable_exif.get("Model")
    datetime_taken = readable_exif.get("DateTimeOriginal") or readable_exif.get("DateTime")

    return {
        "has_exif": True,
        "camera_make": camera_make,
        "camera_model": camera_model,
        "datetime_original": datetime_taken,
        "note": "Presence of camera metadata suggests the image likely originated from a physical camera, but EXIF can be stripped or edited, so this is supporting evidence only, not proof."
    }
