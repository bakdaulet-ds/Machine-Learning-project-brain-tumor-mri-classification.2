import argparse
from imutils import paths

from dataset.DatasetLoader import DatasetLoader
from preprocessing.ToGray import ToGray
from preprocessing.AspectAwarePreprocess import AspectAwarePreprocess


ap = argparse.ArgumentParser()
ap.add_argument(
    "-d", "--dataset",
    required=True,
    help="Path to dataset folder"
)
ap.add_argument(
    "-o", "--output",
    default="save",
    help="Output HDF5 filename without extension"
)
args = vars(ap.parse_args())

imagePaths = list(paths.list_images(args["dataset"]))

print(f"[INFO] Found {len(imagePaths)} images")

aap = AspectAwarePreprocess(32, 32)
tg = ToGray()

dl = DatasetLoader(
    hdf_file=args["output"],
    preprocessors=[aap, tg]
)

dl.load(imagePaths, verbose=1000)

print("[INFO] HDF5 file created successfully")
