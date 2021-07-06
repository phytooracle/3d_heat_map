#!/usr/bin/env python3
"""
Author : Ariyan Zarei, Emmanuel Gonzalez
Date   : 2021-07-02
Purpose: Generate heatmap from pointcloud
"""

import argparse
import os
import sys
from utilities import *
import matplotlib.cm
import json


# --------------------------------------------------
def get_args():
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description='Generate heatmap from pointcloud',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-d',
                        '--downsampled_pointcloud',
                        help='Downsampled pointcloud.',
                        metavar='str',
                        type=str,
                        required=True)

    parser.add_argument('-t',
                        '--tiff',
                        help='Orthomosaic in TIFF format.',
                        metavar='str',
                        type=str,
                        nargs='+',
                        required=True)
    
    parser.add_argument('-o',
                        '--out_dir',
                        help='Heatmap output directory.',
                        metavar='str',
                        type=str,
                        default='heatmap_out')

    return parser.parse_args()


# --------------------------------------------------
def create_out_path(out_path):

    if not os.path.isdir(out_path):
        os.makedirs(out_path)


# --------------------------------------------------
def main():
    """Make a jazz noise here"""

    args = get_args()
    result_json = {}
    basename = os.path.splitext(os.path.basename(args.downsampled_pointcloud))[0]
    create_out_path(args.out_dir)

    # Load previously generated downsampled pointcloud.
    downsampled_pcd = load_pcd(args.downsampled_pointcloud)

    # Open orthomosaic and collect coordinates.
    main_ortho = load_full_ortho(args.tiff[0])
    ortho_coords = get_coord_from_tiff(args.tiff[0])

    # Generate heatmap from the downsampled pointcloud.
    pcd_image_main,pcd_bounds,pcd_bounds_utm = generate_height_image_from_pcd(downsampled_pcd,main_ortho,ortho_coords)
    cv2.imwrite(''.join([os.path.join(args.out_dir, basename), '_heatmap.png']), pcd_image_main)

    result_json = {'pcd_bounds': pcd_bounds,
                   'pcd_bounds_utm': pcd_bounds_utm}

    with open(''.join([os.path.join(args.out_dir, basename), '.json']), 'w') as outfile:
        json.dump(result_json, outfile)


# --------------------------------------------------
if __name__ == '__main__':
    main()
