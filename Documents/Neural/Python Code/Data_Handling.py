# note, check if co_ordinates start at 0,0 or 1,1 and fix get_pixel_list method in generate_output_file
# note, if bounding box overlap is possible, then add compression to pixel_list santization function
import matplotlib.pylab as plt

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
    def get_bounding_boxes():
        # from an input pixel list file, generate a dictionary of bounding boxes
        bounding_boxes = {}
        for obj in data_handler.object_list:
            bounding_boxes[obj] = []

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

# test for generate_output_file method
# generates an empty bounding box dictionary then attempts to print the output file
bboxes = data_handler.get_bounding_boxes()
print(data_handler.generate_output_file(bboxes, "img_000.jpg"))