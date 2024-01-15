import glob
import os
import shutil
from urllib.parse import unquote, urlparse

import numpy as np
import supervisely as sly
from cv2 import connectedComponents
from dataset_tools.convert import unpack_if_archive
from supervisely.io.fs import file_exists, get_file_name, get_file_name_with_ext
from tqdm import tqdm

import src.settings as s


def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(
            desc=f"Downloading '{file_name_with_ext}' to buffer...",
            total=fsize,
            unit="B",
            unit_scale=True,
        ) as pbar:
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path


def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    ### Function should read local dataset and upload it to Supervisely project, then return project info.###
    images_path = "/home/alex/DATASETS/TODO/vocalfolds/img"
    masks_path = "/home/alex/DATASETS/TODO/vocalfolds/annot"
    ds_name = "ds"
    batch_size = 30
    images_folder = "/img/"
    masks_folder = "/annot/"

    def create_ann(image_path):
        labels = []
        tags = []

        subfolder_value = image_path.split("/")[-2]
        curr_meta = seq_to_tag[subfolder_value]
        subfolder = sly.Tag(curr_meta)
        tags.append(subfolder)

        patient_value = image_path.split("/")[-3]
        patient_meta = patient_to_meta[patient_value]
        patient = sly.Tag(patient_meta)
        tags.append(patient)

        img_height = 512
        img_wight = 512

        mask_path = image_path.replace(images_folder, masks_folder)

        if file_exists(mask_path):
            mask_np = sly.imaging.image.read(mask_path)[:, :, 0]
            unique_pixels = np.unique(mask_np)

            for pixel in unique_pixels:
                obj_class = pixel_to_class.get(pixel)
                mask = mask_np == pixel
                curr_bitmap = sly.Bitmap(mask)
                curr_label = sly.Label(curr_bitmap, obj_class)
                labels.append(curr_label)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels, img_tags=tags)

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)

    seq1_meta = sly.TagMeta("seq1", sly.TagValueType.NONE)
    seq2_meta = sly.TagMeta("seq2", sly.TagValueType.NONE)
    seq3_meta = sly.TagMeta("seq3", sly.TagValueType.NONE)
    seq4_meta = sly.TagMeta("seq4", sly.TagValueType.NONE)
    seq5_meta = sly.TagMeta("seq5", sly.TagValueType.NONE)
    seq6_meta = sly.TagMeta("seq6", sly.TagValueType.NONE)
    seq7_meta = sly.TagMeta("seq7", sly.TagValueType.NONE)
    seq8_meta = sly.TagMeta("seq8", sly.TagValueType.NONE)
    patient1_meta = sly.TagMeta("patient 1", sly.TagValueType.NONE)
    patient2_meta = sly.TagMeta("patient 2", sly.TagValueType.NONE)
    patient_to_meta = {"patient1": patient1_meta, "patient2": patient2_meta}

    seq_to_tag = {
        "seq1": seq1_meta,
        "seq2": seq2_meta,
        "seq3": seq3_meta,
        "seq4": seq4_meta,
        "seq5": seq5_meta,
        "seq6": seq6_meta,
        "seq7": seq7_meta,
        "seq8": seq8_meta,
    }

    pixel_to_class = {
        0: sly.ObjClass("void", sly.Bitmap, color=(128, 128, 128)),
        1: sly.ObjClass("vocal folds", sly.Bitmap, color=(255, 0, 0)),
        2: sly.ObjClass("other tissue", sly.Bitmap, color=(0, 0, 255)),
        3: sly.ObjClass("glottal space", sly.Bitmap, color=(0, 128, 0)),
        4: sly.ObjClass("pathology", sly.Bitmap, color=(128, 0, 128)),
        5: sly.ObjClass("surgical tool", sly.Bitmap, color=(255, 165, 0)),
        6: sly.ObjClass("intubation", sly.Bitmap, color=(255, 255, 0)),
    }

    meta = sly.ProjectMeta(
        tag_metas=[
            patient1_meta,
            patient2_meta,
            seq1_meta,
            seq2_meta,
            seq3_meta,
            seq4_meta,
            seq5_meta,
            seq6_meta,
            seq7_meta,
            seq8_meta,
        ],
        obj_classes=list(pixel_to_class.values()),
    )

    api.project.update_meta(project.id, meta.to_json())

    dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

    images_pathes = glob.glob(images_path + "/*/*/*.png")

    progress = sly.Progress("Create dataset {}".format(ds_name), len(images_pathes))

    for img_pathes_batch in sly.batched(images_pathes, batch_size=batch_size):
        img_names_batch = [
            im_path.split("/")[-2] + "_" + get_file_name_with_ext(im_path)
            for im_path in img_pathes_batch
        ]

        img_infos = api.image.upload_paths(dataset.id, img_names_batch, img_pathes_batch)
        img_ids = [im_info.id for im_info in img_infos]

        anns = [create_ann(image_path) for image_path in img_pathes_batch]
        api.annotation.upload_anns(img_ids, anns)

        progress.iters_done_report(len(img_names_batch))

    return project
