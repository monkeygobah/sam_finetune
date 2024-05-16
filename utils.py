import os
import cv2
from scipy import stats
import numpy as np
# Set the paths to your image and mask directories
from patchify import patchify  #Only to handle large images
import matplotlib.pyplot as plt
import random
import ast

import csv


# def get_modes(image_dir):
#     # Get a list of image file names
#     image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg") or f.endswith(".png")]

#     # Initialize lists to store image dimensions
#     widths = []
#     heights = []

#     # Iterate over the image files and get their dimensions
#     for image_file in image_files:
#         image_path = os.path.join(image_dir, image_file)
#         image = cv2.imread(image_path)
#         height, width = image.shape[:2]
#         widths.append(width)
#         heights.append(height)

#     # Calculate the mode width and height
#     mode_width = int(stats.mode(widths, keepdims=True).mode[0])
#     mode_height = int(stats.mode(heights, keepdims = True).mode[0])
#     return mode_width, mode_height
    
    
def jitter_box(box, jitter_range=20):
    x_min, y_min, x_max, y_max = box
    
    # Generate random jitter values within the specified range
    x_jitter = random.randint(-jitter_range, jitter_range)
    y_jitter = random.randint(-jitter_range, jitter_range)
    
    # Apply jitter to the box coordinates
    x_min += x_jitter
    y_min += y_jitter
    x_max += x_jitter
    y_max += y_jitter
    
    return [float(x_min), float(y_min), float(x_max), float(y_max)]

def split_and_resize(image, mask):
    height, width = image.shape[:2]
    midpoint = width // 2

    right_image = image[:, :midpoint]
    left_image = image[:, midpoint:]

    right_image = cv2.resize(right_image, (256, 256), interpolation=cv2.INTER_NEAREST)
    left_image = cv2.resize(left_image, (256, 256), interpolation=cv2.INTER_NEAREST)

    right_label = mask[:, :midpoint]
    left_label = mask[:, midpoint:]

    right_label = cv2.resize(right_label, (256, 256), interpolation=cv2.INTER_NEAREST)
    left_label = cv2.resize(left_label, (256, 256), interpolation=cv2.INTER_NEAREST)


    return left_image, right_image, left_label, right_label


def get_split_labels_and_boxes(label_key_dict, names, scale_width, scale_height, row,mask, resized_images, adjusted_boxes, resized_masks, resized_image, image_name,  num_jitters=5, jitter_range=10, dir='r'):
    if dir == 'l':
        label_list = ['L_sclera', 'L_brow', 'L_iris']
    else:
        label_list = ['R_sclera', 'R_brow', 'R_iris']
        
    for structure in label_list:
        box_key = f"{structure}_box"
        if box_key in row:
            box = row[box_key]
            box = np.fromstring(box.strip('[]'), sep=(' '))
            adjusted_box = [float(coord * scale_width) if idx % 2 == 0 else float(coord * scale_height)
                            for idx, coord in enumerate(box)]

            # Generate jittered boxes
            jittered_boxes = [adjusted_box]
            for _ in range(num_jitters - 1):
                jittered_box = jitter_box(adjusted_box, jitter_range)
                jittered_boxes.append(jittered_box)

            for jittered_box in jittered_boxes:
                # Create separate mask for each structure
                structure_mask = np.zeros_like(mask)
                structure_index =  label_key_dict[structure]
                structure_mask[mask == structure_index] = 255
                # plt.imshow(structure_mask)
                # plt.savefig('checking_out')
                if structure == 'L_sclera':
                    l_iris_index = label_key_dict['L_iris']
                    l_iris_mask_temp = np.zeros_like(mask)
                    l_iris_mask_temp[mask == l_iris_index ] = 255
                    
                    l_sclera_mask = cv2.bitwise_or(structure_mask, l_iris_mask_temp)
                    resized_masks.append(l_sclera_mask)

                elif structure == 'R_sclera':
                    
                    r_iris_index = label_key_dict['R_iris']
                    
                    r_iris_mask_temp = np.zeros_like(mask)
                    r_iris_mask_temp[mask ==r_iris_index ] = 255
                    
                    r_sclera_mask = cv2.bitwise_or(structure_mask, r_iris_mask_temp)
                    resized_masks.append(r_sclera_mask)
                    
                else:
                    # save_mask = cv2.resize(structure_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
                    resized_masks.append(structure_mask)


                resized_images.append(resized_image)
                adjusted_boxes.append(jittered_box)
                names.append(image_name[:-4])
                
    return resized_masks, resized_images, adjusted_boxes, names



def resize_image_and_boxes(csv_file, image_dir, mask_dir):
    resized_images = []
    resized_masks = []
    adjusted_boxes = []
    names = []
    structures = []

    # label_list = ['L_sclera', 'R_sclera', 'L_brow', 'R_brow','L_iris', 'R_iris' ]
    label_key_dict = {
        'L_sclera': 1 ,
        'R_sclera': 2,
        'L_brow': 3,
        'R_brow':4,
        'L_iris':5,
        'R_iris':6
    }
    with open(csv_file, 'r') as file:
        csv_reader = csv.DictReader(file)
        for idx, row in enumerate(csv_reader):
            image_name = row['new_crop_name']
            
            try:
                image = cv2.imread(os.path.join(image_dir, image_name))
                mask_name = image_name[:-4] + '.png'
                mask = cv2.imread(os.path.join(mask_dir, mask_name), cv2.IMREAD_GRAYSCALE)
            
                original_height, original_width = map(int, row['original_size'][1:-1].split(', '))
                left_image, right_image, left_label, right_label= split_and_resize(image, mask)
                                
                scale_width = 256 / original_width
                scale_height = 256 / original_height

                resized_masks, resized_images, adjusted_boxes, names = get_split_labels_and_boxes(label_key_dict, names, scale_width, scale_height, row, left_label, resized_images, adjusted_boxes, \
                                            resized_masks, left_image, image_name,  num_jitters=5, jitter_range=10, dir='l')
                resized_masks, resized_images, adjusted_boxes, names = get_split_labels_and_boxes(label_key_dict, names, scale_width, scale_height, row, right_label, resized_images, adjusted_boxes, \
                                            resized_masks, right_image, image_name,  num_jitters=5, jitter_range=10, dir='r')                    
                    
            except AttributeError:
                pass
                    # # Adjust and jitter bounding boxes
                    # for structure in label_list:
                    #     box_key = f"{structure}_box"
                    #     if box_key in row:
                    #         box = row[box_key]
                    #         box = np.fromstring(box.strip('[]'), sep=(' '))
                    #         adjusted_box = [float(coord * scale_width) if idx % 2 == 0 else float(coord * scale_height)
                    #                         for idx, coord in enumerate(box)]

                    #         # Generate jittered boxes
                    #         jittered_boxes = [adjusted_box]
                    #         for _ in range(num_jitters - 1):
                    #             jittered_box = jitter_box(adjusted_box, jitter_range)
                    #             jittered_boxes.append(jittered_box)

                    #         for jittered_box in jittered_boxes:
                    #             # Create separate mask for each structure
                    #             structure_mask = np.zeros_like(mask)
                    #             structure_index = label_list.index(structure) + 1
                    #             structure_mask[mask == structure_index] = 255

                    #             if structure == 'L_sclera':
                    #                 l_sclera_mask_temp = cv2.resize(structure_mask, (mode_width, mode_height), interpolation=cv2.INTER_NEAREST)
                    #                 l_iris_index = label_list.index('L_iris') + 1
                    #                 l_iris_mask_temp = np.zeros_like(mask)
                    #                 l_iris_mask_temp[mask == l_iris_index ] = 255
                                    
                    #                 l_iris_mask_temp =  cv2.resize(l_iris_mask_temp, (mode_width, mode_height), interpolation=cv2.INTER_NEAREST)
                                    
                    #                 l_sclera_mask = cv2.bitwise_or(l_sclera_mask_temp, l_iris_mask_temp)
                    #                 resized_masks.append(l_sclera_mask)

                    #             elif structure == 'R_sclera':
                    #                 r_sclera_mask_temp = cv2.resize(structure_mask, (mode_width, mode_height), interpolation=cv2.INTER_NEAREST)
                                    
                    #                 r_iris_index = label_list.index('R_iris') + 1
                    #                 r_iris_mask_temp = np.zeros_like(mask)
                    #                 r_iris_mask_temp[mask ==r_iris_index ] = 255
                                    
                    #                 r_iris_mask_temp =  cv2.resize(r_iris_mask_temp, (mode_width, mode_height), interpolation=cv2.INTER_NEAREST)
                                    
                    #                 r_sclera_mask = cv2.bitwise_or(r_sclera_mask_temp, r_iris_mask_temp)
                    #                 resized_masks.append(r_sclera_mask)
                                    
                    #             else:
                    #                 save_mask = cv2.resize(structure_mask, (mode_width, mode_height), interpolation=cv2.INTER_NEAREST)
                    #                 resized_masks.append(save_mask)


                    #             resized_images.append(resized_image)
                    #             adjusted_boxes.append(jittered_box)
                    #             names.append(image_name[:-4])

 

                # except Exception as e:
                #     pass
    return np.array(resized_images), np.array(resized_masks), adjusted_boxes, names

































import matplotlib.pyplot as plt

def plot_sample_img(resized_images, dataset, image_index=0):
    # Get the unique image names
    unique_names = list(set(dataset["names"]))

    # Select the desired image name
    image_name = unique_names[image_index]

    # Get the indices of all entries corresponding to the selected image
    image_indices = [i for i, name in enumerate(dataset["names"]) if name == image_name]

    # Get the number of structures for the selected image
    num_structures = len(image_indices)

    # Create a figure with subplots
    fig, axs = plt.subplots(2, num_structures, figsize=(20, 10))

    # Plot the image and masks for each structure
    for i, idx in enumerate(image_indices):
        image = resized_images[idx]
        mask = dataset["label"][idx]

        # Plot the image
        axs[0, i].imshow(image)
        axs[0, i].set_title(f"Image: {image_name}")
        axs[0, i].axis("off")

        # Plot the mask
        axs[1, i].imshow(mask, cmap="gray")
        axs[1, i].axis("off")

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Display the plot
    plt.savefig('sample_image_mask_ALL.jpg')
