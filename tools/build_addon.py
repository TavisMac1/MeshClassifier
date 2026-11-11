"""Package the Blender add-on as an installable ZIP."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
ADDON_DIR = ROOT / "blender_addon"
SERVICE_DIR = ROOT / "service"
OUTPUT = ROOT / "dist" / "mesh_classifier.zip"


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(OUTPUT, "w", ZIP_DEFLATED) as archive:
        for path in ADDON_DIR.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                archive.write(path, Path("mesh_classifier") / path.relative_to(ADDON_DIR))
        for path in SERVICE_DIR.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                archive.write(
                    path,
                    Path("mesh_classifier") / "runtime_service" / path.relative_to(SERVICE_DIR),
                )
    print("Created %s" % OUTPUT)


if __name__ == "__main__":
    main()
