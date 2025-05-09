import os

classes = [
    "defense", "eiffel", "general", "invalides", "louvre",
    "moulinrouge", "museedorsay", "notredame", "pantheon",
    "pompidou", "sacrecoeur", "triomphe"
]

input_file = "Paris/list_of_images.txt"
output_file = "Paris/list_of_images_fixed.txt"

with open(input_file, "r") as fin, open(output_file, "w") as fout:
    for line in fin:
        for cls in classes:
            line = line.replace(f"images/{cls}/", "images/")
            line = line.replace(f"{cls}/", "")  # <- extra por si faltaba "images/"
        fout.write(line)

print(f"✅ Archivo corregido guardado como: {output_file}")
