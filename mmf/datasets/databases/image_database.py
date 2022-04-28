# Copyright (c) Facebook, Inc. and its affiliates.
import collections
import os

import torch
import torchvision
import torchvision.datasets.folder as tv_helpers
from mmf.utils.file_io import PathManager
from mmf.utils.general import get_absolute_path
from PIL import Image


def get_possible_image_paths(path):
    image_path = path.split(".")
    # Image path might contain file extension (e.g. .jpg),
    # In this case, we want the path without the extension
    image_path = image_path if len(image_path) == 1 else image_path[:-1]
    for ext in tv_helpers.IMG_EXTENSIONS:
        image_ext = ".".join(image_path) + ext
        if PathManager.isfile(image_ext):
            path = image_ext
            break
    return path


def default_loader(path):
    with PathManager.open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class ImageDatabase(torch.utils.data.Dataset):
    """ImageDatabase可以用来加载MMF中的图像。
   这可以和AnnotationDatabase一起使用，也可以单独使用`from_path`等函数。
   如果`use_images`为True，MMFDataset会初始化自己的ImageDatabase副本。
   如果你把annotation_db作为一个参数，其余的一切工作都与普通的torch数据集一样。
   例如，对于标注数据库中的item 1，你可以向ImageDatabase传递相同的ID来加载其图像。
   如果你不传递它，你有两个选择。要么使用.get来获取标注数据库中的项目，要么使用.from_path来直接获取图像路径。
   你可以自由地使用你自己的数据集而不是图像数据库，或者自由地更新或忽略MMFDataset的ImageDataset初始化。
   你可以用转换和其他参数重新初始化，或者使用Torchvision的任何数据集。
    """

    def __init__(
        self,
        config,
        path,
        annotation_db=None,
        transform=None,
        loader=default_loader,
        is_valid_file=None,
        image_key=None,
        *args,
        **kwargs
    ):
        """Initialize an instance of ImageDatabase

        Args:
            torch ([type]): [description]
            config (DictConfig): Config object from dataset_config
            path (str): 图片数据集的路径，多个路径用逗号分隔
            annotation_db (AnnotationDB, optional): Annotation DB to be used
                to be figure out image paths. Defaults to None.
            transform (callable, optional): Transform to be called upon loaded image.
                Defaults to None.
            loader (callable, optional): Custom loader for image which given a path
                returns a PIL Image. Defaults to torchvision's default loader.
            is_valid_file (callable, optional): Custom callable to filter out invalid
                files. If image is invalid, {"images": []} will returned which you can
                filter out in your dataset. Defaults to None.
            image_key (str, optional): Key that points to image path in annotation db.
                If not specified, ImageDatabase will make some intelligent guesses
                about the possible key. Defaults to None.
        """
        super().__init__()
        self.config = config
        self.base_path = get_absolute_path(path)
        self.transform = transform
        self.annotation_db = annotation_db
        self.loader = loader
        self.image_key = config.get("image_key", None)
        self.image_key = image_key if image_key else self.image_key
        self.is_valid_file = is_valid_file

    @property
    def annotation_db(self):
        return self._annotation_db

    @annotation_db.setter
    def annotation_db(self, annotation_db):
        self._annotation_db = annotation_db

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, transform):
        if isinstance(transform, collections.abc.MutableSequence):
            transform = torchvision.Compose(transform)
        self._transform = transform

    def __len__(self):
        self._check_annotation_db_present()
        return len(self.annotation_db)

    def __getitem__(self, idx):
        self._check_annotation_db_present()
        item = self.annotation_db[idx]
        return self.get(item)

    def _check_annotation_db_present(self):
        if not self.annotation_db:
            raise AttributeError(
                "'annotation_db' must be set for the database to use __getitem__."
                + " Use image_database.annotation_db to set it."
            )

    def get(self, item):
        possible_images = self._get_attrs(item)
        return self.from_path(possible_images)

    def from_path(self, paths, use_transforms=True):
        if isinstance(paths, str):
            paths = [paths]

        assert isinstance(
            paths, collections.abc.Iterable
        ), "Path needs to a string or an iterable"

        loaded_images = []
        for image in paths:
            image = os.path.join(self.base_path, image)
            path = get_possible_image_paths(image)

            valid = self.is_valid_file(path) if self.is_valid_file is not None else True

            if not valid:
                continue

            if not path:
                # Create the full path without extension so it can be printed
                # for the error
                possible_path = ".".join(image.split(".")[:-1])

                raise RuntimeError(
                    "Image not found at path {}.{{jpeg|jpg|svg|png}}.".format(
                        possible_path
                    )
                )
            image = self.open_image(path)

            if self.transform and use_transforms:
                image = self.transform(image)
            loaded_images.append(image)

        return {"images": loaded_images}

    def open_image(self, path):
        return self.loader(path)

    def _get_attrs(self, item):
        """Returns possible attribute that can point to image id

        Args:
            item (Object): Object from the DB

        Returns:
            List[str]: List of possible images that will be copied later
        """
        if self.image_key:
            image = item[self.image_key]
            if isinstance(image, str):
                image = [image]
            return image

        image = None
        pick = None
        attrs = self._get_possible_attrs()

        for attr in attrs:
            image = item.get(attr, None)
            if image is not None:
                pick = attr
                break

        if pick == "identifier" and "left_url" in item and "right_url" in item:
            return [image + "-img0", image + "-img1"]
        else:
            return [image]

    def _get_possible_attrs(self):
        return [
            "Flickr30kID",
            "Flikr30kID",
            "identifier",
            "image_path",
            "image_name",
            "img",
            "image_id",
        ]
