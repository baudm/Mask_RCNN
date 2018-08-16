#!/usr/bin/env python3

import multiprocessing
from PIL import Image
import numpy as np
import os.path

GT = 'panoptic_train2017_pixelmaps'
P = 'segmentations/masks'


def work(proc_id, stuff, images, num_classes, rev_map):
    cm = np.zeros((num_classes, num_classes), dtype=np.float32)
    included = 0
    for working_idx, fname in enumerate(images):
        if working_idx % 100 == 0:
            print('Core: {}, {} from {} images converted'.format(proc_id, working_idx, len(images)))

        img_gt = np.asarray(Image.open(os.path.join(GT, fname))).copy()
        img_p = np.asarray(Image.open(os.path.join(P, fname)))

        if img_gt.shape != img_p.shape:
            print('Shape mismatch:', img_gt.shape, img_p.shape)
            continue

        # Filter thing classes
        img_gt[img_gt < stuff[1]] = 0

        labels = np.unique(img_p)
        included += 1
        # Remove void label
        # labels = labels[labels > 0]
        for label in labels:
            mask = (img_p == label)
            classes, counts = np.unique(img_gt[mask], return_counts=True)
            counts = counts / counts.sum()
            # stats = cm.setdefault(label, {})
            for cls, cnt, in zip(classes, counts):
                cm[rev_map[label]][rev_map[cls]] += cnt
    cm = cm / included
    return cm


def main():
    stuff = [0, 92, 93, 95, 100, 107, 109, 112, 118, 119, 122, 125, 128, 130, 133, 138, 141, 144, 145, 147, 148, 149, 151, 154, 155, 156, 159, 161, 166, 168, 171, 175, 176, 177, 178, 180, 181, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200]
    num_classes = len(stuff)
    rev_map = dict(zip(stuff, range(num_classes)))

    images = os.listdir(GT)

    cpu_num = multiprocessing.cpu_count()
    images_split = np.array_split(images, cpu_num)
    print("Number of cores: {}, images per core: {}".format(cpu_num, len(images_split[0])))
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []
    for proc_id, image_set in enumerate(images_split):
        p = workers.apply_async(work,
                                (proc_id, stuff, image_set, num_classes, rev_map))
        processes.append(p)
    annotations = []
    for p in processes:
        annotations.append(p.get())


    # annotations = np.cat(annotations, dim=-1).mean(dim=-1)
    annotations = np.dstack(annotations).mean(axis=-1)


    np.savez_compressed('conf-matrix.npz', cm=annotations)

    print(annotations)



if __name__ == '__main__':
    main()