from pathlib import Path
import shutil
import re
import time

src = Path(r"D:\IQA\data_base_path\IQA\New_old_roi_c7c8c9_5")
dst = Path(r"D:\IQA\data_base_path\IQA\New_old_roi_c7c8c9_5_numeric_only")
exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

print("src exists:", src.exists(), src)
print("dst:", dst)

scanned = 0
copied = 0
t0 = time.time()

for p in src.rglob("*"):
    scanned += 1

    if scanned % 1000 == 0:
        print(f"scanned={scanned}, copied={copied}, elapsed={time.time()-t0:.1f}s", flush=True)

    if not p.is_file():
        continue
    if p.suffix.lower() not in exts:
        continue

    # 只保留纯数字命名，比如 1.jpg、002.png、123.bmp
    if not re.fullmatch(r"\d+", p.stem):
        continue

    target = dst / p.relative_to(src)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, target)
    copied += 1

print("done")
print("scanned", scanned)
print("copied", copied)
print("to:", dst)
