import os
import xml.etree.ElementTree as ET
import math
import cv2 as cv
import argparse
from tqdm import tqdm

# Class names
classes = ["car", "truck", "bus", "van", "freight car"]


# Define path arguments
def parse_args():
    parser = argparse.ArgumentParser(description='polygon')
    parser.add_argument('--in_xml_vi_dir', default='/home/data1/yanght/DroneVehicle/train/trainlabel', help='RGB XML file directory')
    parser.add_argument('--in_xml_ir_dir', default='/home/data1/yanght/DroneVehicle/train/trainlabelr', help='IR XML file directory')
    parser.add_argument('--out_vi_txt_dir', default='/home/data1/yanght/DroneVehicle/train/trainlabel', help='RGB TXT file directory')
    parser.add_argument('--out_ir_txt_dir', default='/home/data1/yanght/DroneVehicle/train/trainlabelr', help='IR TXT file directory')
    parser.add_argument('--in_vi_img_dir', default='/home/data1/yanght/DroneVehicle/train/trainimg', help='Cropped RGB image directory')
    parser.add_argument('--in_ir_img_dir', default='/home/data1/yanght/DroneVehicle/train/trainimgr', help='Cropped IR image directory')

    # parser.add_argument('--in_xml_vi_dir', default='/home/data1/yanght/DroneVehicle/val/vallabel', help='RGB XML file directory')
    # parser.add_argument('--in_xml_ir_dir', default='/home/data1/yanght/DroneVehicle/val/vallabelr', help='IR XML file directory')
    # parser.add_argument('--out_vi_txt_dir', default='/home/data1/yanght/DroneVehicle/val/vallabel', help='RGB TXT file directory')
    # parser.add_argument('--out_ir_txt_dir', default='/home/data1/yanght/DroneVehicle/val/vallabelr', help='IR TXT file directory')
    # parser.add_argument('--in_vi_img_dir', default='/home/data1/yanght/DroneVehicle/val/valimg', help='Cropped RGB image directory')
    # parser.add_argument('--in_ir_img_dir', default='/home/data1/yanght/DroneVehicle/val/valimgr', help='Cropped IR image directory')

    # parser.add_argument('--in_xml_vi_dir', default='/home/data1/yanght/DroneVehicle/test/testlabel', help='RGB XML file directory')
    # parser.add_argument('--in_xml_ir_dir', default='/home/data1/yanght/DroneVehicle/test/testlabelr', help='IR XML file directory')
    # parser.add_argument('--out_vi_txt_dir', default='/home/data1/yanght/DroneVehicle/test/testlabel', help='RGB TXT file directory')
    # parser.add_argument('--out_ir_txt_dir', default='/home/data1/yanght/DroneVehicle/test/testlabelr', help='IR TXT file directory')
    # parser.add_argument('--in_vi_img_dir', default='/home/data1/yanght/DroneVehicle/test/testimg', help='Cropped RGB image directory')
    # parser.add_argument('--in_ir_img_dir', default='/home/data1/yanght/DroneVehicle/test/testimgr', help='Cropped IR image directory')

    args = parser.parse_args()
    return args


def drow_polygon(in_xml_dir, out_txt_dir, in_img_dir):
    """Draw polygon annotations"""
    if not os.path.exists(out_txt_dir):
        os.mkdir(out_txt_dir)

    for obj in tqdm(os.listdir(in_xml_dir)):
        print(obj)

        tree = ET.parse(os.path.join(in_xml_dir, obj))

        root = tree.getroot()

        size = root.find("size")

        width = size.find("width").text
        height = size.find("height").text
        # depth = size.find("depth").text

        file_path = ""
        img_path = os.path.join(in_img_dir, obj[:-4] + ".jpg")

        image_quality = 1
        result = {}

        if os.path.exists(img_path):
            file_path = img_path
            image_quality = 1
        else:
            file_path = os.path.join(in_img_dir, obj[:-4] + ".png")

            if os.path.exists(file_path):
                image_quality = 1
            else:
                image_quality = 0

        with open(os.path.join(out_txt_dir, obj[:-4] + ".txt"), 'w') as f_dst:
            for obj in root.findall("object"):
                name = obj.find("name").text

                # difficult = obj.find("difficult").text

                if name not in classes:
                    continue

                if name in classes:
                    id = classes.index(name)

                polygon = obj.find("polygon")

                if polygon is None:
                    continue

                points = polygon.findall("point")

                if points is None:
                    continue

                pts = [[0 for _ in range(2)] for _ in range(4)]

                for i, point in enumerate(points):
                    x = float(point.text.split(",")[0])
                    y = float(point.text.split(",")[1])
                    pts[i] = [x, y]

                p1, p2, p3, p4 = pts
                cx = (p1[0] + p2[0] + p3[0] + p4[0]) / 4.0
                cy = (p1[1] + p2[1] + p3[1] + p4[1]) / 4.0

                if image_quality == 0:
                    continue

                file_path = ""

                w = int(p2[0] - p1[0])
                h = int(p4[1] - p3[1])

                cvimg = cv.imread(file_path)

                cvimg_crop = cvimg[int(cy) - h // 2:int(cy) + h // 2, int(cx) - w // 2:int(cx) + w // 2]

                if 0 in cvimg_crop.shape:
                    continue

                x1 = (p1[0] - (cx - w // 2)) / w
                y1 = (p1[1] - (cy - h // 2)) / h

                x2 = (p2[0] - (cx - w // 2)) / w
                y2 = (p2[1] - (cy - h // 2)) / h

                x3 = (p3[0] - (cx - w // 2)) / w
                y3 = (p3[1] - (cy - h // 2)) / h

                x4 = (p4[0] - (cx - w // 2)) / w
                y4 = (p4[1] - (cy - h // 2)) / h

                f_dst.write(str(id) + " " + str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2) + " " + str(x3) + " " + str(y3) + " " + str(x4) + " " + str(y4) + "\n")


if __name__ == '__main__':
    args = parse_args()

    print("RGB Training Data...")
    drow_polygon(args.in_xml_vi_dir, args.out_vi_txt_dir, args.in_vi_img_dir)
    print("IR Training Data...")
    drow_polygon(args.in_xml_ir_dir, args.out_ir_txt_dir, args.in_ir_img_dir)
