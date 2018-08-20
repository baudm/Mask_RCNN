"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights. Also auto download COCO dataset
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet --download=True

    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last

    # Run COCO evaluatoin on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""

import os
import sys
import time
import numpy as np
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)
import json
import shutil

from PIL import Image
import skimage.io as sio
# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import zipfile
import urllib.request
import shutil

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = "2017"

############################################################
#  Configurations
############################################################


class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes

    # BG + thing categories (lumped as one) + original stuff categories + merged stuff categories
    NUM_CLASSES_PANOPTIC = 1 + 1 + 36 + 17


############################################################
#  Dataset
############################################################

class CocoDataset(utils.Dataset):
    def load_coco(self, dataset_dir, subset, year=DEFAULT_DATASET_YEAR, class_ids=None,
                  class_map=None, return_coco=False, auto_download=False, panoptic=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """

        if auto_download is True:
            self.auto_download(dataset_dir, subset, year)

        annotations = "{}/annotations/panoptic_{}{}_instances.json" if panoptic else "{}/annotations/instances_{}{}.json"

        if subset == 'test-dev':
            annotations = "{}/annotations/image_info_{}-dev{}.json"
            subset = 'test'
        elif subset == 'test':
            annotations = "{}/annotations/image_info_{}{}.json"

        coco = COCO(annotations.format(dataset_dir, subset, year))
        if subset == "minival" or subset == "valminusminival":
            subset = "val"
        image_dir = "{}/images/{}{}".format(dataset_dir, subset, year)

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        # if class_ids and subset != 'test':
        #     image_ids = []
        #     for id in class_ids:
        #         image_ids.extend(list(coco.getImgIds(catIds=[id])))
        #     # Remove duplicates
        #     image_ids = list(set(image_ids))
        # else:
        # All images
        image_ids = list(coco.imgs.keys())

        # Add classes
        if panoptic:
            import json
            dir = os.path.dirname(os.path.realpath(__file__))
            with open(os.path.join(dir, 'panoptic_coco_categories.json'), 'r') as f:
                self.panoptic_coco_categories = json.load(f)
            things = []
            stuff = []
            for c in self.panoptic_coco_categories:
                cat = things if c['isthing'] else stuff
                cat.append(c['id'])
            things.sort()
            stuff.sort()
            self.things_category_mapping = dict(zip(things, range(1, len(things) + 1)))
            self.things_category_rev = {v: k for k, v in self.things_category_mapping.items()}
            self.things_category_mapping[0] = 0
            self.things_category_rev[0] = 0

            self.stuff_category_mapping = dict(zip(stuff, range(2, len(stuff) + 2)))
            self.stuff_category_rev = {v: k for k, v in self.stuff_category_mapping.items()}
            self.stuff_category_mapping[0] = 0
            self.stuff_category_rev[0] = 0
            self.stuff_category_rev[1] = 1
            # Map all thing categories to class '1'
            for c in things:
                self.stuff_category_mapping[c] = 1

        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        path_to_semantic = "{}/annotations/panoptic_{}{}_pixelmaps".format(dataset_dir, subset, year)

        # Add images
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)),
                path_semantic=os.path.join(path_to_semantic, "{:012d}.png".format(i)))

        if return_coco:
            return coco

    def auto_download(self, dataDir, dataType, dataYear):
        """Download the COCO dataset/annotations if requested.
        dataDir: The root directory of the COCO dataset.
        dataType: What to load (train, val, minival, valminusminival)
        dataYear: What dataset year to load (2014, 2017) as a string, not an integer
        Note:
            For 2014, use "train", "val", "minival", or "valminusminival"
            For 2017, only "train" and "val" annotations are available
        """

        # Setup paths and file names
        if dataType == "minival" or dataType == "valminusminival":
            imgDir = "{}/{}{}".format(dataDir, "val", dataYear)
            imgZipFile = "{}/{}{}.zip".format(dataDir, "val", dataYear)
            imgURL = "http://images.cocodataset.org/zips/{}{}.zip".format("val", dataYear)
        else:
            imgDir = "{}/{}{}".format(dataDir, dataType, dataYear)
            imgZipFile = "{}/{}{}.zip".format(dataDir, dataType, dataYear)
            imgURL = "http://images.cocodataset.org/zips/{}{}.zip".format(dataType, dataYear)
        # print("Image paths:"); print(imgDir); print(imgZipFile); print(imgURL)

        # Create main folder if it doesn't exist yet
        if not os.path.exists(dataDir):
            os.makedirs(dataDir)

        # Download images if not available locally
        if not os.path.exists(imgDir):
            os.makedirs(imgDir)
            print("Downloading images to " + imgZipFile + " ...")
            with urllib.request.urlopen(imgURL) as resp, open(imgZipFile, 'wb') as out:
                shutil.copyfileobj(resp, out)
            print("... done downloading.")
            print("Unzipping " + imgZipFile)
            with zipfile.ZipFile(imgZipFile, "r") as zip_ref:
                zip_ref.extractall(dataDir)
            print("... done unzipping")
        print("Will use images in " + imgDir)

        # Setup annotations data paths
        annDir = "{}/annotations".format(dataDir)
        if dataType == "minival":
            annZipFile = "{}/instances_minival2014.json.zip".format(dataDir)
            annFile = "{}/instances_minival2014.json".format(annDir)
            annURL = "https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip?dl=0"
            unZipDir = annDir
        elif dataType == "valminusminival":
            annZipFile = "{}/instances_valminusminival2014.json.zip".format(dataDir)
            annFile = "{}/instances_valminusminival2014.json".format(annDir)
            annURL = "https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip?dl=0"
            unZipDir = annDir
        else:
            annZipFile = "{}/annotations_trainval{}.zip".format(dataDir, dataYear)
            annFile = "{}/instances_{}{}.json".format(annDir, dataType, dataYear)
            annURL = "http://images.cocodataset.org/annotations/annotations_trainval{}.zip".format(dataYear)
            unZipDir = dataDir
        # print("Annotations paths:"); print(annDir); print(annFile); print(annZipFile); print(annURL)

        # Download annotations if not available locally
        if not os.path.exists(annDir):
            os.makedirs(annDir)
        if not os.path.exists(annFile):
            if not os.path.exists(annZipFile):
                print("Downloading zipped annotations to " + annZipFile + " ...")
                with urllib.request.urlopen(annURL) as resp, open(annZipFile, 'wb') as out:
                    shutil.copyfileobj(resp, out)
                print("... done downloading.")
            print("Unzipping " + annZipFile)
            with zipfile.ZipFile(annZipFile, "r") as zip_ref:
                zip_ref.extractall(unZipDir)
            print("... done unzipping")
        print("Will use annotations in " + annFile)

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(CocoDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(CocoDataset, self).load_mask(image_id)

    def load_sem_mask(self,image_id):
        """Load semantic masks for the given image.

        Returns:
        masks: A bool array of shape [height, width] with
            mask identified per pixel
        class_ids: a 1D array of class IDs of the instance masks. (FOR REVIEW)
        """

        """ TO DO: specify function in parent class. IF still needed """
        # Call the mask in a specific path (annotations/semantic_val)
        sem_image = sio.imread(self.image_info[image_id]["path_semantic"])
        # If RGB. Convert to grayscale for class labels.
        if sem_image.ndim != 2 :
            raise("labels should not be in RGB format")

        # Map COCO class IDs to local (sequential) class IDs
        sem_image = np.vectorize(self.stuff_category_mapping.get)(sem_image)

        sem_image = np.expand_dims(sem_image, axis=2)

        return sem_image

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(CocoDataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


############################################################
#  COCO Evaluation
############################################################

def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None,panoptic_path=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # load the annotations file
    dataDir='/media/airscan/Disk1/MS-COCO/2017'
    dataType='val2017'
    #annFile='{}/{}/annotations/panoptic_{}_instances.json'.format(dataDir,dataType,dataType)
    annFile='{}/{}/annotations/panoptic_{}.json'.format(dataDir,dataType,dataType)

    #with open('panoptic_coco_categories.json') as json_data:
    #    panoptic_categories = json.load(json_data)

    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    # for panoptic
    sem_images = []
    masks = []
    class_ids = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"],
                                           r["masks"].astype(np.uint8))
        # sort the class ids and given masks
        results.extend(image_results)
        sem_images.append(r['sem_mask'])
        masks.append(r['masks'])
        class_ids.append(r['class_ids'])

    if panoptic_path:
        # Outputs folders and json
        output_json = os.path.join(panoptic_path,'panoptic_instances.json')
        # create the json file for panoptic: format_converter.py
        gt_json = os.path.join(panoptic_path,'panoptic_gt.json')
        create_info_json(output_json, coco_image_ids,annFile,gt_json)

        ch2_path = os.path.join(panoptic_path,'ch2_folder')
        if os.path.isdir(ch2_path):
            shutil.rmtree(ch2_path)
        input_path = os.path.join(panoptic_path,'comparison_folder')
        if os.path.isdir(input_path):
            shutil.rmtree(input_path)
        gt_path = os.path.join(panoptic_path,'gt_folder')
        if os.path.isdir(gt_path):
            shutil.rmtree(gt_path)
        # create new directory
        os.makedirs(ch2_path)
        os.makedirs(gt_path)
        os.makedirs(input_path)
        # list of 2-channel images
        ch2_images = []
        for i,image in enumerate(sem_images):
            ch2_image = get_2ch_image(dataset,image,masks[i],class_ids[i])
            ch2_images.append(ch2_image)

            # get the name of the image
            image_name = '{:012d}.png'.format(coco_image_ids[i])
            # save the the image
            out_img = Image.fromarray(ch2_image.astype(np.uint8))
            out_img.save(os.path.join(ch2_path,image_name))
            # "input image"
            input_img = dataset.load_image(image_ids[i])
            comb_img = np.zeros((input_img.shape[0],input_img.shape[1]*2,3))
            comb_img[:,:input_img.shape[1],:] = ch2_image
            comb_img[:,input_img.shape[1]:,:] = input_img
            img = Image.fromarray(comb_img.astype(np.uint8))
            img.save(os.path.join(input_path,'view_'+image_name))
            # "ground_truth image"
            gt_img_path = '/media/airscan/Disk1/MS-COCO/2017/val2017/annotations/panoptic_val2017' # change to folder path of ground truth
            gt_img_path = os.path.join(gt_img_path,image_name)
            gt_img = sio.imread(gt_img_path)
            gt_img = Image.fromarray(gt_img.astype(np.uint8))
            gt_img.save(os.path.join(gt_path,image_name))
            print('Image {:05} done saving'.format(i),end='\r')

    # Load results. This modifies results with additional attributes.
    #coco_results = coco.loadRes(results)

    # Evaluate
    #cocoEval = COCOeval(coco, coco_results, eval_type)
    #cocoEval.params.imgIds = coco_image_ids
    #cocoEval.evaluate()
    #cocoEval.accumulate()
    #cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)

############################################################
#  Panoptic Functions
############################################################
def create_info_json(output_json_path,image_ids,anno_json,gt_json_path=None):
    with open(anno_json) as json_data:
        images_info = json.load(json_data)

    d_coco = images_info.copy() # copy images_info
    d_coco_gt = images_info.copy()
    del d_coco['annotations']

    d_coco['images'] = []
    d_coco_gt['images'] = []
    d_coco_gt['annotations'] = []

    for image_info in images_info['images']:
        if image_info['id'] in image_ids:
            d_coco['images'].append(image_info)
            d_coco_gt['images'].append(image_info)

    for anno_info in images_info['annotations']:
        if anno_info['image_id'] in image_ids:
            d_coco_gt['annotations'].append(anno_info)

    with open(output_json_path,'w') as json_data:
        json.dump(d_coco,json_data)

    if gt_json_path:
        with open(gt_json_path,'w') as json_data:
            json.dump(d_coco_gt,json_data)


def create_info_json2(output_json_path,image_ids):
    obj = {'images': [{
                'id': int(image_id),
                'file_name': '{:012d}.jpg'.format(int(image_id))
                } for image_id in image_ids]}
    with open(output_json_path, 'w') as f:
        json.dump(obj, f)


def s_to_p_classes(dataset,mask):
    """ inputs a semantic mask with labels sorted as arange ~[0,138] ann
    returns a semantic mask with desired panoptic labels """

    # convert each pixel to Panoptic class
    flat_mask = mask.flatten()
    flat_pmask = np.zeros(flat_mask.shape)

    pmask = np.vectorize(dataset.stuff_category_rev.get)(mask)
    #for i,px in enumerate(flat_mask):
    #    if px > 0: # errors if ID is 0
    #        flat_pmask[i] = dataset.things_category_rev[px]

    # reshape to original mask size
    #pmask = flat_pmask.reshape(mask.shape)

    return pmask

def get_2ch_image(dataset,sem_image,masks,class_ids):
    """override the semantic mask labels in the 1st channel
                 count the instance of masks in 2nd channel
       dataset: CocoDataset instance
       sem_image: The semantic image output
       masks: mask of the predicted image (h,w,n), n = number of objects in image
       class_ids: classes present in the image, should be aligned with masks arrangement
    """

    # convert all present ids to source_class_id
    h,w = (sem_image.shape[0],sem_image.shape[1])    # save in np.zeros(height,width,3)
    ch2_image = np.zeros((h,w,3),dtype='uint8')

    psem_image = s_to_p_classes(dataset,sem_image)
    ch2_image[:,:,0] = psem_image            # store the semantic labels

    try:
        source_ids = np.vectorize(dataset.things_category_rev.get)(class_ids)
    except ValueError:
        source_ids = []

    mask_sizes = np.count_nonzero(masks, axis=(0, 1))

    ## IDEA: sort classes by mean mask size first (biggest - background),
    #        then sort masks within the same class (biggest - foreground)
    # However, doesn't work as well as simple sorting of masks
    # unique_classes = np.unique(source_ids)
    # mean_mask_sizes = [mask_sizes[source_ids == c].mean() for c in unique_classes]
    # mean_mask_map = dict(zip(unique_classes, mean_mask_sizes))

    # Sort per class first (classes with high mean mask size most probably are "background" objects)
    # Then, per class, sort individual masks: masks with low mask size are probably farther away from the camera.
    # Meanwhile, within the same class, masks with higher mask size are probably closer to the camera.
    # This implements some sort of depth ordering
    #order = sorted(range(len(source_ids)), key=lambda i: (-mean_mask_map[source_ids[i]], mask_sizes[i]))

    order = np.argsort(mask_sizes)
    # Start from biggest mask to smallest
    order = np.flip(order, axis=-1)
    # create the 2ch PNG
    unique_ids = dict()
    for i in order:
        id = source_ids[i]
        if id not in unique_ids:
            unique_ids[id] = 0  # instance count per id
        else:
            unique_ids[id] += 1
        y,x = np.where(masks[:,:,i]==1)
        ch2_image[y,x,0] = id # prioritize the class of the instance
        ch2_image[y,x,1] = unique_ids[id]

    return ch2_image


def get_category_name(id=None):
    # load the panoptic categories -> from panoptic id
    with open('panoptic_coco_categories.json') as json_data:
        panoptic_categories = json.load(json_data)

    if id:
        for cat in panoptic_categories:
            if cat['id'] == id:
                return cat['name']
    else:
        return 'void'

############################################################
#  Training
############################################################

def generate_submission(model, dataset_path, panoptic_path):
    from PIL import Image
    import numpy as np
    import os.path

    t_prediction = 0
    t_start = time.time()

    dataset = CocoDataset()
    dataset.load_coco(dataset_path, 'test', year='2017', panoptic=True)
    dataset.prepare()

    image_ids = dataset.image_ids
    print(len(image_ids))
    print(dataset.class_names)

    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    ch2_path = os.path.join(panoptic_path, 'ch2_folder')
    if os.path.isdir(ch2_path):
        shutil.rmtree(ch2_path)
    # create new directory
    os.makedirs(ch2_path)

    for i, image_id in enumerate(image_ids):

        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        try:
            r = model.detect([image], verbose=0)[0]
        except ValueError as e:
            print('error:', image.shape, e)
            continue
        t_prediction += (time.time() - t)

        # Ignore the semantic mask prediction for now, since DeepLabv3 output is better
        #sem_mask = r['sem_mask']
        sem_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        ch2_image = get_2ch_image(dataset, sem_mask, r['masks'], r['class_ids'])

        # get the name of the image
        image_name = '{:012d}.png'.format(coco_image_ids[i])
        # save the the image
        out_img = Image.fromarray(ch2_image.astype(np.uint8))
        out_img.save(os.path.join(ch2_path, image_name))
        # "input image"
        # "ground_truth image"

        print('Image {:05d} done saving'.format(i))

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on MS COCO")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--year', required=False,
                        default=DEFAULT_DATASET_YEAR,
                        metavar="<year>",
                        help='Year of the MS-COCO dataset (2014 or 2017) (default=2017)')
    parser.add_argument('--panoptic', required=False,
                        default=True,
                        metavar="<True|False>",
                        help='Use panoptic dataset or not (default=True)',
                        type=bool)
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    parser.add_argument('--download', required=False,
                        default=False,
                        metavar="<True|False>",
                        help='Automatically download and unzip MS-COCO files (default=False)',
                        type=bool)
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Year: ", args.year)
    print("Panoptic: ", args.panoptic)
    print("Logs: ", args.logs)
    print("Auto Download: ", args.download)

    # Configurations
    if args.command == "train":
        config = CocoConfig()
    else:
        class InferenceConfig(CocoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = CocoDataset()
        dataset_train.load_coco(args.dataset, "train", year=args.year, auto_download=args.download, panoptic=args.panoptic)
        if args.year in '2014':
            assert not args.panoptic
            dataset_train.load_coco(args.dataset, "valminusminival", year=args.year, auto_download=args.download)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = CocoDataset()
        val_type = "val" if args.year in '2017' else "minival"
        dataset_val.load_coco(args.dataset, val_type, year=args.year, auto_download=args.download, panoptic=args.panoptic)
        dataset_val.prepare()

        # Image Augmentation
        # Right/Left flip 50% of the time
        augmentation = imgaug.augmenters.Fliplr(0.5)

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 0
        print("Training semantic segmentation head")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='semantic',
                    augmentation=augmentation)

        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='heads',
                    augmentation=augmentation)

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=120,
                    layers='4+',
                    augmentation=augmentation)

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=160,
                    layers='all',
                    augmentation=augmentation)

    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = CocoDataset()
        val_type = "val" if args.year in '2017' else "minival"
        coco = dataset_val.load_coco(args.dataset, val_type, year=args.year, return_coco=True, auto_download=args.download, panoptic=args.panoptic)
        dataset_val.prepare()

        panoptic_output_path = None
        if args.panoptic:
            panoptic_output_path = os.path.join(args.dataset,'panoptic_output')
            if os.path.isdir(panoptic_output_path):
                shutil.rmtree(panoptic_output_path) # Warning: deletes the current panoptic output folder
            os.makedirs(panoptic_output_path)

        print("Running COCO evaluation on {} images.".format(args.limit))
        evaluate_coco(model, dataset_val, coco, "bbox", limit=int(args.limit), panoptic_path=panoptic_output_path)

    elif args.command == 'submission':
        panoptic_output_path = os.path.join(args.dataset, 'panoptic_output')
        generate_submission(model, args.dataset, panoptic_path=panoptic_output_path)


    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
