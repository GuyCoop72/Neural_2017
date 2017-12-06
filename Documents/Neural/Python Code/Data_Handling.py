# note, check if co_ordinates start at 0,0 or 1,1 and fix get_pixel_list method in generate_output_file
# note, if bounding box overlap is possible, then add compression to pixel_list santization function
import matplotlib.pylab as plt
import os
import numpy as np

class data_handler:
    image_size = [400, 400]

    object_list = [
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor"
    ]

    def __init__(self, _data_dir):
        self.data_dir = _data_dir
        self.train_dir = _data_dir + "/train/"
        self.test_dir = _data_dir + "/test/"
        return


    @staticmethod
    def generate_output_file(bounding_boxes, filename):
        #generate an output pixel list from the bounding boxes dictionary
        def get_one_pixel_list(bounding_box):
            # generates a list of pixels for a rectangular bounding box
            pixel_list = []
            h_run = bounding_box[1][0] - bounding_box[0][0]
            for j in range(bounding_box[0][1], bounding_box[1][1]):
                # perform vertical steps down
                pixel_list.append([bounding_box[0][0] + (data_handler.image_size[1] * j), h_run])
            return pixel_list

        def sort_pixel_list(pixel_list):
            # sorts pixel list and manages possible overlaps
            pixel_list = sorted(pixel_list, key=lambda x:x[0])

            #check if overlap is possible, if so perform compressions stage
            return pixel_list

        def str_pixel_list(pixel_list):
            # convert a pixel list into a space seperated string
            pix_string = ""
            for pair in pixel_list:
                pix_string += str(pair[0]) + " " + str(pair[1]) + " "
            return pix_string

        def get_full_pixel_list(bounding_boxes):
            if bounding_boxes == []:
                return [[0, 1]]

            else:
                pixel_list = []
                for bounding_box in bounding_boxes:
                    pixel_list += get_one_pixel_list(bounding_box)

                pixel_list = sort_pixel_list(pixel_list)
                return pixel_list

        output_string = ""
        for obj in data_handler.object_list:
            output_string = output_string + filename + "_" + obj + ","
            pixel_list = get_full_pixel_list(bounding_boxes[obj])
            output_string += str_pixel_list(pixel_list)
            output_string += '\n'

        return output_string

    @staticmethod
    def get_bounding_boxes(label):
        # from an input pixel list file, generate a dictionary of bounding boxes
        def pixel_list_to_bboxes(pixel_list):
            pixel_loc_corners = []
            bboxes = []
            #start pixel location, current left pixel location, run length
            if pixel_list[0][0] == 1 and pixel_list[0][1] == 0:
                return bounding_boxes
            pixel_loc_corners.append([pixel_list[0][0], pixel_list[0][0], pixel_list[0][1]])
            for pair in pixel_list[1:]:
                break_var = 0
                for ind, objs_found in enumerate(pixel_loc_corners):
                    if objs_found[1] + data_handler.image_size[1] == pair[0]:
                        # not a new object
                        pixel_loc_corners[ind][1] = pair[0]
                        break_var = 1
                        break
                if break_var == 0:
                    pixel_loc_corners.append([pair[0], pair[0], pair[1]])

            # finished building pixel_loc_corners
            for corner in pixel_loc_corners:
                x1 = ((corner[0]) % data_handler.image_size[0]) + 1
                y1 = int(corner[0] / data_handler.image_size[1])
                x2 = ((corner[1] + corner[2] - 1) % data_handler.image_size[0]) + 1
                y2 = int((corner[1] + corner[2]) / data_handler.image_size[1])
                bboxes.append([[x1, y1],[x2, y2]])

            return bboxes


        pixel_list_dict = {}
        for ind, line in enumerate(label.readlines()):
            pixel_list_dict[data_handler.object_list[ind]] = map(int, (line.split(',')[1].split()))
            pixel_list_dict[data_handler.object_list[ind]] = np.reshape(
                pixel_list_dict[data_handler.object_list[ind]],
                [len(pixel_list_dict[data_handler.object_list[ind]])/2, 2])

        bounding_boxes = {}
        for obj in data_handler.object_list:
            bounding_boxes[obj] = pixel_list_to_bboxes(pixel_list_dict[obj])

        return bounding_boxes

    @staticmethod
    def get_sub_image(image_array, pixel_list):
        # gets a flattened sub_image from a full image and a list of pixels.
        output_array = []
        for pair in pixel_list:
            for i in range(pair[0], pair[0] + pair[1]):
                output_array.append(image_array[i])
        return output_array

    @staticmethod
    def image_read(file_name):
        # reads a jpg file into a [x * y, 3] array containing rgb values
        plt.imread(file_name)
        output_array = np.reshape()

    @staticmethod
    def get_yolo_text_files(input_file_location, output_file_location):
        output_string = ""
        f = open(input_file_location)
        bboxes = data_handler.get_bounding_boxes(f)
        for obj in data_handler.object_list:
            if bboxes[obj] != bboxes:
                for bbox in bboxes[obj]:
                    output_string += (obj + " " \
                                     + str(bbox[0][0]) + " " \
                                     + str(bbox[0][1]) + " " \
                                     + str(bbox[1][0] - bbox[0][0]) + " " \
                                     + str(bbox[1][1] - bbox[0][1]) + '\n')

        outfile = open(output_file_location, 'w')
        outfile.write(output_string)
        outfile.close()
        return output_string

# test for generate_output_file method
# generates an empty bounding box dictionary then attempts to print the output file
bboxes = data_handler.get_bounding_boxes(open("/home/guy/Documents/Neural/Data/train/2007_000042.txt"))
print(data_handler.get_yolo_text_files("/home/guy/Documents/Neural/Data/train/2007_000042.txt"))
