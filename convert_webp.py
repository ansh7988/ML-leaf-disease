from PIL import Image
from pathlib import Path

folder = Path("data/raw/healthy")

for file in folder.glob("*.webp"):
    img = Image.open(file).convert("RGB")

    new_file = file.with_suffix(".jpg")
    img.save(new_file, "JPEG")

    print(f"Converted: {file.name} -> {new_file.name}")

print("Conversion completed!")