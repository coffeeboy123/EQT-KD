from __future__ import absolute_import, division, print_function

import os
import os.path as osp
import re
from typing import List, Tuple

from .. import ImageDataset


class RGBNT201(ImageDataset):
    """RGBNT201 image Re-ID dataset loader.

    Expected directory layout::

        <root>/
            RGBNT201/
                bounding_box_train/
                bounding_box_validation/  # optional for legacy splits
                bounding_box_test/
                query/

    Expected filename::

        000013_cam3_1_00.jpg
        | PID  |cam|seq|frame|

    Only PID and camera ID are required by the current Re-ID engine. Sequence
    and frame fields are parsed for filename validation but are not returned.
    """

    dataset_dir = "RGBNT201"
    _junk_pids = []

    _filename_pattern = re.compile(
        r"^(?P<pid>\d+)_cam(?P<camid>\d+)_"
        r"(?P<sequence>\d+)_(?P<frame>\d+)"
        r"\.(?P<extension>jpg|jpeg|png|bmp)$",
        flags=re.IGNORECASE,
    )
    _image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(self, root="", **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.data_dir = osp.join(self.root, self.dataset_dir)

        self.train_dir = osp.join(self.data_dir, "bounding_box_train")
        self.validation_dir = osp.join(
            self.data_dir, "bounding_box_validation"
        )
        self.query_dir = osp.join(self.data_dir, "query")
        self.gallery_dir = osp.join(self.data_dir, "bounding_box_test")

        self.check_before_run(
            [
                self.data_dir,
                self.train_dir,
                self.query_dir,
                self.gallery_dir,
            ]
        )

        train = self.process_dir(self.train_dir, relabel=True)

        # The A/B protocol has an independent validation directory. Legacy
        # RGBNT201 splits without one fall back to the training list so the
        # project's mandatory validation loader can still be constructed.
        if osp.isdir(self.validation_dir):
            validation = self.process_dir(
                self.validation_dir, relabel=True
            )
        else:
            validation = list(train)

        # This custom A/B protocol deliberately evaluates same-camera temporal
        # retrieval because most train_171 identities occur in only one camera.
        # Assigning distinct protocol camera IDs prevents the project's
        # Market1501 evaluator from discarding valid same-camera positives.
        query = self.process_dir(
            self.query_dir,
            relabel=False,
            protocol_camid=0,
        )
        gallery = self.process_dir(
            self.gallery_dir,
            relabel=False,
            protocol_camid=1,
        )

        self._check_evaluation_split(query, gallery)

        super(RGBNT201, self).__init__(
            train,
            validation,
            query,
            gallery,
            **kwargs
        )

    @classmethod
    def _parse_filename(cls, filename: str) -> Tuple[int, int]:
        match = cls._filename_pattern.fullmatch(filename)
        if match is None:
            raise ValueError(
                "Unexpected RGBNT201 filename: {!r}. Expected a name such as "
                "'000013_cam3_1_00.jpg'.".format(filename)
            )

        pid = int(match.group("pid"))
        camera_number = int(match.group("camid"))
        if camera_number < 1:
            raise ValueError(
                "Camera IDs must start at 1, but got {!r}.".format(filename)
            )

        # The project represents camera IDs with zero-based integers.
        camid = camera_number - 1
        return pid, camid

    @classmethod
    def _list_images(cls, directory: str) -> List[str]:
        image_paths = []
        for entry in os.scandir(directory):
            if not entry.is_file():
                continue
            extension = osp.splitext(entry.name)[1].lower()
            if extension in cls._image_extensions:
                image_paths.append(entry.path)
        return sorted(image_paths)

    def process_dir(
        self,
        directory: str,
        relabel: bool = False,
        protocol_camid=None,
    ):
        image_paths = self._list_images(directory)
        if not image_paths:
            raise RuntimeError(
                "No supported images were found in {!r}.".format(directory)
            )

        parsed = [
            (image_path, *self._parse_filename(osp.basename(image_path)))
            for image_path in image_paths
        ]

        pid2label = {}
        if relabel:
            original_pids = sorted({pid for _, pid, _ in parsed})
            pid2label = {
                original_pid: label
                for label, original_pid in enumerate(original_pids)
            }

        data = []
        for image_path, pid, camid in parsed:
            if relabel:
                pid = pid2label[pid]
            if protocol_camid is not None:
                camid = int(protocol_camid)
            data.append((image_path, pid, camid))
        return data

    @staticmethod
    def _check_evaluation_split(query, gallery) -> None:
        query_pids = {pid for _, pid, _ in query}
        gallery_pids = {pid for _, pid, _ in gallery}
        missing_pids = sorted(query_pids - gallery_pids)
        if missing_pids:
            preview = missing_pids[:10]
            raise RuntimeError(
                "Query identities missing from gallery: {}{}".format(
                    preview,
                    " ..." if len(missing_pids) > len(preview) else "",
                )
            )
