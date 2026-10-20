"""Подготовка данных: скачивание Yambda-50M и KION, проверка контрольных сумм, сборка единых таблиц событий.

Источники:
  Yambda (Яндекс, Apache-2.0): https://huggingface.co/datasets/yandex/yambda   (подвыборка 50M: flat/50m и маппинги артист/альбом)
  KION (МТС): https://github.com/irsafilo/KION_DATASET   (лицензия в репозитории не указана; используется для учебного исследования)
Запуск: python scripts/00_prepare_data.py
"""
import hashlib
import urllib.request
import zipfile
from pathlib import Path

from recsys.config import DATA
from recsys.data import kion, yambda

HF = "https://huggingface.co/datasets/yandex/yambda/resolve/main/"
GH = "https://github.com/irsafilo/KION_DATASET/raw/main/"

# файл -> (адрес, sha256)
YAMBDA_FILES = {
    "listens.parquet": ("flat/50m/listens.parquet", "eed9cbd094af1e189507d2f8132a0dc9653b90e65480125c7cdccd601e0592d1"),
    "likes.parquet": ("flat/50m/likes.parquet", "694087077dbacfcc1d22a5ca85cc6bd8ab182361933406e60500624c6a422bc4"),
    "dislikes.parquet": ("flat/50m/dislikes.parquet", "64f240a6d62b8acdfa6efae660a54d10ebc64f9f2f0aeeef8e6bb0dbe7dc2802"),
    "unlikes.parquet": ("flat/50m/unlikes.parquet", "532213992934de765874691ac145324078e61987df26c88e1d671d23f8853180"),
    "undislikes.parquet": ("flat/50m/undislikes.parquet", "8cdbcf6e7e79491d560637057ba8ac078dd1610a3fab15442b6f12e45069ef85"),
    "artist_item_mapping.parquet": ("artist_item_mapping.parquet", "cf1110aa2413ed70008ec9232d395660b2181ffda731002fb74da54e87b6f25d"),
    "album_item_mapping.parquet": ("album_item_mapping.parquet", "559b6576063d96280414574b32292eee7b47ee0b9f2952cb3ae670a5d19125e4"),
}
KION_ZIP = ("data_en.zip", "fddc5b2215bfff49bcfa21e464441f1fc6dc30f8f4afe41d3f23bf3fc45f3542")
# отдельный interactions.csv.zip в репозитории обрезан (ровно 50 МиБ), поэтому берётся только data_en.zip


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, path: Path, expected: str) -> None:
    if path.exists() and sha256(path) == expected:
        print(f"  {path.name}: уже скачан, контрольная сумма верна")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  скачиваю {url}")
    urllib.request.urlretrieve(url, path)
    assert sha256(path) == expected, f"{path.name}: контрольная сумма не совпала"


def safe_extract(archive: Path, target: Path, members: tuple[str, ...]) -> None:
    """Распаковывает только нужные файлы, не допуская выхода за пределы каталога."""
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            name = info.filename
            if name.startswith(("/", "\\")) or ".." in Path(name).parts or not name.startswith(members):
                continue
            z.extract(info, target)


print("Yambda-50M")
for name, (path, digest) in YAMBDA_FILES.items():
    fetch(HF + path, yambda.RAW / name, digest)
print("KION")
zip_path = DATA / "raw" / "kion_zip" / KION_ZIP[0]
fetch(GH + KION_ZIP[0], zip_path, KION_ZIP[1])
safe_extract(zip_path, DATA / "raw" / "kion", ("data_en/",))
print("Сборка таблиц событий")
yambda.build()
kion.build()
print("готово")
