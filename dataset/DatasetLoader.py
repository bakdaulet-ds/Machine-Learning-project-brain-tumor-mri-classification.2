import cv2
import h5py


class DatasetLoader:
    def __init__(self, hdf_file, preprocessors=None):
        self.hdf_file = hdf_file
        self.preprocessors = preprocessors if preprocessors is not None else []

    def _extract_label(self, imagePath):
        parts = imagePath.replace("\\", "/").split("/")
        return parts[-3]  # glioma / meningioma / notumor / pituitary

    def load(self, imagePaths, verbose=500):
        path = self.hdf_file + ".hdf5"

        first_image = cv2.imread(imagePaths[0])
        if first_image is None:
            raise ValueError(f"Could not read first image: {imagePaths[0]}")

        for p in self.preprocessors:
            first_image = p.preprocess(first_image)

        h, w = first_image.shape[:2]
        dt_lbl = h5py.string_dtype(encoding="utf-8")

        with h5py.File(path, "w") as f:
            images = f.create_dataset(
                "images",
                shape=(len(imagePaths), h, w),
                dtype="uint8",
                compression="gzip",
                compression_opts=5,
                shuffle=True
            )

            labels = f.create_dataset(
                "labels",
                shape=(len(imagePaths),),
                dtype=dt_lbl
            )

            for i, imagePath in enumerate(imagePaths):
                image = cv2.imread(imagePath)

                if image is None:
                    print(f"[WARNING] Could not read image: {imagePath}")
                    continue

                label = self._extract_label(imagePath)

                for p in self.preprocessors:
                    image = p.preprocess(image)

                images[i] = image
                labels[i] = label

                if verbose > 0 and (i + 1) % verbose == 0:
                    print(f"[INFO] Processed {i + 1}/{len(imagePaths)} | label={label}")

        return True
