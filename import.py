from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, Region
from msrest.authentication import ApiKeyCredentials
import time
import os
import json
import gc
import pandas as pd
import argparse
from tqdm import tqdm
import math
tqdm.pandas()


def reformat_coco_categories(coco_cats):
    new_coco_cats = {}
    for cat in coco_cats:
        new_coco_cats[cat["id"]] = cat["name"]
    return new_coco_cats


def coco_to_df(coco):
    df_image = pd.json_normalize(coco["images"]).set_index("id")
    df_annots = pd.json_normalize(coco["annotations"]).set_index("id")
    df = df_annots.join(df_image, on="image_id", how="left", rsuffix="_image")
    df = df.rename({"file_name": "image"}, axis=1)
    normalization_flag = df_annots["area"].max() > 1
    temp = pd.DataFrame(
        list(df.apply(lambda x: bbox_reformat(x, normalization_flag), axis=1)))

    temp.index = df.index
    df = df.join(temp, rsuffix="_bbox")
    return df


def bbox_reformat(row, normalization_flag):
    if normalization_flag:
        bbox = {
            "x": row["bbox"][0] / float(row["width"]),
            "y": row["bbox"][1] / float(row["height"]),
            "width": row["bbox"][2] / float(row["width"]),
            "height": row["bbox"][3] / float(row["height"])
        }
    else:
        bbox = {
            "x": row["bbox"][0],
            "y": row["bbox"][1],
            "width": row["bbox"][2],
            "height": row["bbox"][3]
        }
    return bbox

def paginate_dataframe(dataframe, page_size, page_num):

    page_size = page_size

    if page_size is None:

        return None

    offset = page_size*(page_num-1)

    return dataframe[offset:offset + page_size]

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--project_id',        
        default='', #update this with the new project id
        type=str,
        help='Name of project to create in Custom Vision'
    )
    parser.add_argument(
        '--endpoint',
        default='',
        type=str,
        help='URL to the Custom Vision Endpoint'
    )
    parser.add_argument(
        '--api_key',
        default='',
        type=str,
        help='API Key for Custom Vision'
    )
    parser.add_argument(
        '--image_directory',
        default='',  #Update this path for the data set
        type=str,
        help='Path to the training data images'
    )
    parser.add_argument(
        '--coco_file_path',
        default='',  #Update this path for the data set
        type=str,
        help='Path to the COCO annotations JSON file'
    )
    parser.add_argument(
        '--upload_batch_size',
        type=str,        
        default=64,
        help='Number of images to upload in a batch'
    )
    args = parser.parse_args()
    
    assert os.path.isdir(args.image_directory), "Can't find image directory"
    assert os.path.isfile(args.coco_file_path), "Can't find COCO file"
    assert 1 <= args.upload_batch_size <= 64, "Upload batch size must be on or between 1 and 64"

    with open(args.coco_file_path, "r") as f:
        coco = json.load(f)
    # with open("../datasets/synthetic_ds6_2_coco_annotations.json", "r") as f:
    #     alt_coco = json.load(f)
    coco_categories = reformat_coco_categories(coco["categories"])
    df_coco = coco_to_df(coco)
    # df_alt_coco = coco_to_df(alt_coco)
    # df_coco = pd.concat([df_coco, df_alt_coco])
    df_coco["category"] = df_coco["category_id"].apply(
        lambda x: coco_categories[x])
    credentials = ApiKeyCredentials(in_headers={"Training-key": args.api_key})
    trainer = CustomVisionTrainingClient(args.endpoint, credentials)
    model_types = pd.DataFrame(
        [{"id": x.id, "name": x.name, "type": x.type} for x in trainer.get_domains()])
    obj_detection_domain = next(domain for domain in trainer.get_domains(
    ) if domain.type == "ObjectDetection" and domain.name == "General (compact) [S1]")
    
    print("Adding new dataset to existing project id {0}".format(args.project_id))
    project = trainer.get_project(project_id=args.project_id)   # existing project
    trainer_tags = trainer.get_tags(project.id)
    tags = {}
    for trainer_tag in trainer_tags:
        tags[trainer_tag.name] = trainer_tag


    def image_tagger(row):

        if row["category"] == 'front' or row["category"] == 'top':
            x, y, w, h = row["x"], row["y"], row["width_bbox"], row["height_bbox"]
            #region = Region(tag_id='f4428727-1923-498f-a360-892bca142b08', #tags[row["category"]].id,

            region = Region(tag_id=tags[row["category"]].id,
                            left=x, top=y, width=w, height=h)
            with open(os.path.join(args.image_directory, row["image"]), mode="rb") as image_contents:
                return ImageFileCreateEntry(name=row["image"], contents=image_contents.read(), regions=[region])
        else:
            return None
    print("Creating tagged image regions")


    page_size = 100
    page_number = 1
    
    while True:
        df = paginate_dataframe(df_coco, page_size, page_number)
        tagged_images_with_regions = list(df.progress_apply(image_tagger, axis=1))


        print(
            f"Uploading in batches of {args.upload_batch_size} ({math.ceil(len(tagged_images_with_regions)/ float(args.upload_batch_size))} batches)")
        for i in tqdm(range(0, len(tagged_images_with_regions), args.upload_batch_size), unit="image batch"):
            upload_result = trainer.create_images_from_files(project.id, ImageFileCreateBatch(
                images=tagged_images_with_regions[i: min((i + args.upload_batch_size), len(tagged_images_with_regions))]))
            if not upload_result.is_batch_successful:
                print("Image batch upload failed.")
                for image in upload_result.images:
                    print("Image status: ", image.status)
        page_number = page_number + 1
        if len(tagged_images_with_regions) < 100:
            break
