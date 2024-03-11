import json
import argparse
from pathlib import Path
import cv2
import datetime
import os
def get_args_parser():
    parser = argparse.ArgumentParser('Set dataset arguments', add_help=False)
    parser.add_argument('--coco_dataset',help="Path to COCO folder")
    parser.add_argument("--description", help="Description of the dataset", default="A custom dataset")
    parser.add_argument("--url", help="URL of the dataset", default="")
    parser.add_argument("--version", help="Version of the dataset", default="1.0")
    parser.add_argument("--contributor", help="Contributor of the dataset", default="Anonymous")
    parser.add_argument("--custom_dataset", help="Path to custom dataset", default="custom_dataset")
    return parser

def get_categories(custom_dataset):
    categorys = []
    with open(Path(custom_dataset) / "classes.txt", "r") as f:
        for category in f:
            categorys.append(category.strip())
    return categorys

def main(args):
    coco_folder = Path(args.coco_dataset)

    for mode in ["train", "val"]:
        new_data = {}
        instance_json = coco_folder / "annotations" /  f"instances_{mode}2017.json"
        # open and load the json file
        with open(instance_json, "r") as f:
            data = json.load(f)
            keys = list(data.keys())
            
            # First key: info
            info = data[keys[0]]
            info_keys = list(info.keys())
            new_data[keys[0]] = {
                "description": args.description,
                "url": args.url,
                "version": args.version,
                "year": datetime.date.today().year,
                "contributor": args.contributor,
                "date_created": datetime.date.today().strftime("%Y-%m-%d")
            }

            # Second key: licenses. Keep it as it is
            # new_data[keys[1]] = data[keys[1]]
            new_data[keys[1]] = {}

            # Third key: images. It is a list with each element as a dictionary for 1 image
            new_data[keys[2]] = list()
            for img_path in (Path(args.custom_dataset) / mode / "images").glob("*.jpg"):
                image_element = {key: [] for key in ['license', 'file_name', 'coco_url', 'height', 'width', 'date_captured', 'flickr_url', 'id']}
                image_element['license'] = 0 # License is 0 for all images
                image_element['file_name'] = img_path.name
                image_element['coco_url'] = "" # No URL for now
                img_height, img_width  = cv2.imread(str(img_path)).shape[:2]
                image_element['height'] = img_height
                image_element['width'] = img_width
                image_element['date_captured'] = datetime.date.today().strftime("%Y-%m-%d")
                image_element['flickr_url'] = "" # No URL for now
                try:
                    image_element['id'] = int(img_path.name[:-4]) # ID is the name of the image without the extension
                except:
                    image_element['id'] = str(img_path.name[:-4]) # test for koala dataset as images is not purely numbers
                new_data[keys[2]].append(image_element)

            # Fourth key: annotations. It is a list with each element as a dictionary for 1 object
            new_data[keys[3]] = list()
            object_id = 1
            for label_path in (Path(args.custom_dataset) / mode / "labels").glob("*.txt"):
                with open(label_path, "r") as f:
                    for line in f:
                        annotation_element = {key: [] for key in ['segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'id']}
                        
                        try:
                            annotation_element['image_id'] = int(label_path.name[:-4] )# ID is the name of the label without the extension
                        except:
                            annotation_element['image_id'] = str(label_path.name[:-4] ) # test for koala dataset as images is not purely numbers
                        image = cv2.imread(str(Path(args.custom_dataset) / mode / "images" / (label_path.name[:-4] + ".jpg")))
                        img_height, img_width  = image.shape[:2]
                        line = line.split()
                        category_id, center_x, center_y, width, height = map(float, line)
                        x1, y1, x2, y2 = (center_x - width/2) * img_width, (center_y - height/2) * img_height, (center_x + width/2) * img_width, (center_y + height/2) * img_height
                        annotation_element['segmentation'] = [] # No segmentation for now
                        annotation_element['area'] = round((x2-x1) * (y2-y1),2)
                        annotation_element['iscrowd'] = 0 # No crowd for now
                        annotation_element['bbox'] = [round(x1,2), round(y1,2), round(x2 - x1, 2), round(y2 - y1,2)]
                        annotation_element['category_id'] = int(category_id) + 1 # ID is the natural number starting from 1
                        annotation_element['id'] = object_id
                        
                        new_data[keys[3]].append(annotation_element)
                        object_id += 1

            # Fifth key: categories. The length of this list is the number of classes
            categorie_names = get_categories(args.custom_dataset)
            category = list()
            for i, cat in enumerate(categorie_names):
                category_element = {key: [] for key in ['supercategory', 'id', 'name']}
                category_element['supercategory'] = cat # keep the supercategory same as the category
                category_element['id'] = i + 1 # ID is the natural number starting from 1
                category_element['name'] = cat
                category.append(category_element)
            new_data[keys[4]] = category

        # print(f"Current new dictionary: \n")
        # print(f"->->->{ new_data['info'] }\n")
        # print(f"->->->{ new_data['licenses'] }\n")
        # print(f"->->->{ new_data['images'] }\n")
        # print(f"->->->{ new_data['annotations'] }\n")
        # print(f"->->->{ new_data['categories'] }\n")

        # save the new json file
        with open(os.path.join(args.custom_dataset, "annotations", f"instances_{mode}2017.json"), "w") as f:
            json.dump(new_data, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Make custom dataset to train DETR', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)