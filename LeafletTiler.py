"""
Usage:  LeafletTiler.py [-h | --help] [FILE] ...
        LeafletTiler.py [--test] [FILE] ...
        LeafletTiler.py [--directory=<targetdir>] colourAndTileTiff (--tiffFilePath=<filepath>) [--newFolderNames=<folders>] [--clean]
        LeafletTiler.py [--directory=<targetdir>] expandTiff (--tiffFilePath=<filepath>) [--colours=<colours>] [--newFolderNames=<folders>]
        LeafletTiler.py [--directory=<targetdir>] changeHue [--colours=<colours>] [FILE] ...
        LeafletTiler.py [--directory=<targetdir>] tile [FILE]
        LeafletTiler.py [--directory=<targetdir>] clean
        LeafletTiler.py [-r | --run]

Options:
  -h --help                   Show this screen.
  -t --test                   Read image files and guess ideal tile sizes and zoomlevels.
  --colours=<colours>         Specify hue value(s) [default: 0] to colour the sepia tile.
  --numfiles=<numfiles>       Specify the number of images in the image stack.
  --tiffFilePath=<filepath>   Specify the filepath of the tiff file to be expanded.
  --newFolderNames=<folders>  Name for folders that will hold the images and tiles from the expanded tiff.
  --directory=<targetdir>     Specify target directory for tiles.
  --clean                     Doesn't save intermediate files.
  -r --run                    Run the application from the index.html file.

Process image FILE (jpg, png, whatever) and assume sane number of zoom levels
"""
import math
import PIL
from docopt import docopt
from PIL import Image, ImageSequence
import numpy as np
import cv2
import sys
import os
import glob
import tifffile
import yaml
from jinja2 import Template
from flask import Flask, render_template, send_from_directory

# defining some fields / global variables:
folder_names = []
img_hues = []
target_dir = '.'
num_files = 0
num_slices = 0
new_dir = os.getcwd()
original_dir = os.getcwd()


def load_config():
    stream = open("config.yaml")
    parsed_yamlfile = yaml.load(stream, Loader=yaml.FullLoader)
    print("\nDEFAULT CONFIG SETTINGS:")
    print("Number of slices for this animal: " + str(parsed_yamlfile["numslices"]))
    print("Number of images per slice: " + str(parsed_yamlfile["numfiles"]))
    layernames_msg = "Image names: {}".format(parsed_yamlfile["layernames"])
    colours_msg = "Image colours: {}".format(parsed_yamlfile["colours"])
    print(layernames_msg)
    print(colours_msg)

    global folder_names
    folder_names = parsed_yamlfile["layernames"]

    global img_hues
    img_hues = parsed_yamlfile["colours"]

    for i in range(int(parsed_yamlfile["numfiles"])):
        print("{0}: {1}".format(folder_names[i], img_hues[i]))

    global num_files
    num_files = parsed_yamlfile["numfiles"]

    global num_slices
    num_slices = parsed_yamlfile["numslices"]

    global original_dir
    original_dir = os.getcwd()
    os.chdir('./templates')

    global new_dir
    new_dir = os.getcwd()  # to be used in html file
    os.chdir(original_dir)

    print("\n")


def add_padding(image, original_width, original_height, directory):

    zoom_levels = get_num_zoom_levels(original_width, original_height)
    factor = (2 ** (zoom_levels - 1)) * 256

    if original_width == original_height:  # i.e. image is square
        if original_width % factor == 0:
            return image
        else:
            new_width = factor * math.ceil(original_width/factor)
            new_height = factor * math.ceil(original_height/factor)

    elif original_width < original_height:  # rectangle w smaller width than height
        if (original_width % factor == 0) & (original_height % factor == 0):
            return image
        else:
            new_width = factor * math.ceil(original_width/factor)
            new_height = math.ceil(original_height/factor) * factor

    else:  # rectangle w smaller height than width
        if (original_width % factor == 0) & (original_height % factor == 0):
            return image
        else:
            new_height = factor * math.ceil(original_height/factor)
            new_width = math.ceil(original_width/factor) * factor

    # create a black rectangle of a size that is a factor of 256*2^n
    w, h = new_width, new_height
    data = np.zeros((h, w, 3), dtype=np.uint8)
    background = Image.fromarray(data, 'RGB')

    # overlay the original image on top of the newly created black square
    top_img = image
    background.paste(top_img, (0, 0))  # (align top left corner of image with background)

    # save the image
    background.save(directory + "/padded_image.jpg")

    return background


def colorize_grayscale(image_filepath, hue):
    """take in greyscale image and apply hue (default 0 which is red)"""
    img_arr = grayscale_to_sepia(image_filepath)
    coloured_img = sepia_to_colour(img_arr, hue)
    processed_img = Image.fromarray(coloured_img)
    return processed_img


def grayscale_to_sepia(image_filepath):
    """adds a sepia hue"""
    img_gray = cv2.imread(image_filepath, cv2.IMREAD_GRAYSCALE)
    img_sepia = cv2.applyColorMap(img_gray, cv2.COLORMAP_OCEAN)
    img_colour = Image.fromarray(img_sepia)
    arr = np.array(img_colour)
    return arr


def sepia_to_colour(arr, hue):
    """shift hue to specified color"""
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)  # OpenCV reads the images as BRG instead of RGB
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hsv[..., 0] = hue
    img_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return img_rgb


def get_num_zoom_levels(img_width, img_height):
    """Read image files and guess ideal num of zoom levels based on tile size of 256x256"""
    if img_width <= img_height:
        zoom_levels = math.ceil(math.log((img_width/256),2)) + 1
    else:
        zoom_levels = math.ceil(math.log((img_height/256),2)) + 1

    return zoom_levels


def tile_image(image_path, xs, ys, zoomlevels, directory):
    tile_width = 256  # square tiles so width = height
    image_width = xs
    image_height = ys

    while zoomlevels > 0:
        os.mkdir(directory + str(zoomlevels) + "/")

        filename = image_path.split("/")[len(image_path.split("/")) - 1]
        cmd = ('convert ' + directory + filename + ' -scale {0}x{1}\! ' + directory + '{2}/full_zoom.png').format(
                                                                                image_width, image_height, zoomlevels)
        print(cmd)
        os.system(cmd)  # to run the command

        cmd2 = ('convert -crop 256X256 +repage ' + directory + '{0}/full_zoom.png ' + directory + '{0}/tiles_%d.png').format(zoomlevels)
        print(cmd2)
        os.system(cmd2)

        fname = glob.glob((directory + "{0}/tiles_*.png").format(zoomlevels))
        total_tiles = len(fname)
        print(total_tiles, zoomlevels)

        tiles_per_column = image_width/tile_width

        row = 0
        column = 0
        for i in range(total_tiles):
            filename = (directory + "{0}/tiles_{1}.png").format(zoomlevels, i)  # current filename
            target = (directory + "{0}/map_{1}_{2}.png").format(zoomlevels, column, row)  # new filename

            print('cp -f {0} {1}'.format(filename, target))
            os.rename(filename, target)

            column = column + 1
            if column >= tiles_per_column:
                column = 0
                row = row + 1

        zoomlevels -= 1
        image_width = image_width // 2
        image_height = image_height // 2

    return True


# this works, just not with ztif files...
# expands tiff files and colours them according to config.yaml file
def expand_tiff(tiff_file):
    print("Started expanding tiff: " + tiff_file)
    tiffstack = Image.open(tiff_file)
    tiffstack.load()
    print("Opened tiff file")
    print("Number of frames: " + str(tiffstack.n_frames))

    global img_hues
    for i in range(tiffstack.n_frames):
        try:
            tiffstack.seek(i)
            os.mkdir(target_dir + "/tiles/" + folder_names[i])
            print("Saving img: " + str(i))
            img_name = target_dir + '/tiles/' + folder_names[i] + '/expanded_%s.jpg' % (i,)
            tiffstack.save(img_name)
            coloured_img = colorize_grayscale(img_name, img_hues[i])
            coloured_img.save(target_dir + '/tiles/' + folder_names[i] + '/color_img_%s.jpg' % (i,))
        except EOFError:
            break


def expand_colour_tile_tiff(tiff_file):
    print("Started expanding tiff: " + tiff_file)
    tiffstack = Image.open(tiff_file)
    tiffstack.load()
    print("Opened tiff file")
    print("Number of frames: " + str(tiffstack.n_frames))
    global target_dir
    target_dir = target_dir + "/tiles/"

    for i in range(tiffstack.n_frames):
        try:
            tiffstack.seek(i)
            os.mkdir(target_dir + folder_names[i])
            print("Saving img: " + str(i))
            img_name = target_dir + folder_names[i] + '/expanded_%s.jpg' % (i,)
            tiffstack.save(img_name)

            # colour tiff based on yaml file OR based on user input
            coloured_img = colorize_grayscale(img_name, img_hues[i])
            coloured_img_name = target_dir + folder_names[i] + '/color_img_%s.jpg' % (i,)
            coloured_img.save(coloured_img_name)

            # add padding to make the image divisible by the tile side length
            saved_coloured_img = PIL.Image.open(coloured_img_name)
            w1, h1 = saved_coloured_img.size
            padded_image = add_padding(saved_coloured_img, w1, h1, target_dir)
            padded_img_name = target_dir + folder_names[i] + '/padded_image.jpg'
            padded_image.save(padded_img_name)
            print("Padded image %s created." % (i,))

            # tile image
            paddedimg_path = target_dir + folder_names[i] + "/padded_image.jpg"
            paddedimg = Image.open(paddedimg_path)
            w2, h2 = paddedimg.size
            zoom_levels = get_num_zoom_levels(w2, h2)
            paddedimg.close()
            tile_image(paddedimg_path, w2, h2, zoom_levels, target_dir + folder_names[i] + '/')

        except EOFError:
            break

    create_white_background(w2, h2, target_dir)


def is_same_size(images):
    file1 = images[0]
    print("opening file: " + file1)
    img1 = PIL.Image.open(file1)
    width1, height1 = img1.size

    for i in images:
        print("opening file: ", i)
        img = PIL.Image.open(i)
        width, height = img.size
        # check if they have the same dimensions
        if (width != width1) | (height != height1):
            print("WARNING: Files are not the same size.")
            return False
        else:
            print("All files are the same size.")
            return True


# background layer so that when images are overlayed and all layers are removed, backgronud white layer remains
def create_white_background(bgd_width, bgd_height, directory):
    white_bgd = np.ones([bgd_height, bgd_width], dtype=np.uint8) * 255
    background = Image.fromarray(white_bgd)
    os.mkdir(directory + "background")
    background.save(directory + "background/background.jpg")

    # tile white background in same directory structure as before
    print("opening file: background.jpg")
    bgd_img = PIL.Image.open(directory + "background/background.jpg")
    zoom_levels = get_num_zoom_levels(bgd_width, bgd_height)
    bgd_img.close()
    tile_image((directory + "background/background.jpg"), bgd_width, bgd_height, zoom_levels, directory + "background/")


def clean():
    for fn in folder_names:
        index = str(folder_names.index(fn))
        print("Removed files: ")
        os.remove('tiles/' + fn + '/expanded_' + index + '.jpg')
        print('tiles/' + fn + '/expanded_' + index + '.jpg')
        os.remove('tiles/' + fn + '/padded_image.jpg')
        print('tiles/' + fn + '/padded_image.jpg')
        os.remove('tiles/' + fn + '/color_img_' + index + '.jpg')
        print('tiles/' + fn + '/color_img_' + index + '.jpg')

    print("Done cleaning")


# run the below only if directly invoked
if __name__ == '__main__':
    arguments = docopt(__doc__)
    load_config()
    print(arguments)

    if arguments['--test']:  # Read image files and guess ideal tile sizes and zoomlevels.
        if arguments['FILE'] is not None:
            is_same_size(arguments['FILE'])
            img = PIL.Image.open(arguments['FILE'][0])
            width, height = img.size
            print("Image size (w, h): (" + str(width) + ", " + str(height) + ")")
            print("Ideal number of zoom levels: " + str(get_num_zoom_levels(width, height)))
            img.close()

    elif arguments['tile']:
        if arguments['--directory'] is not None:
            target_dir = arguments['--directory']
        if arguments['FILE'] is not None:
            if not is_same_size(arguments['FILE']):
                print("WARNING: Files are not the same size.")
            else:
                # tile given image
                for f in arguments['FILE']:
                    print("opening file: ", f)
                    img = PIL.Image.open(f)
                    width, height = img.size
                    padded_img = add_padding(img, width, height, target_dir)  # overlay image onto black background divisible by tile size
                    padded_img.save("padded_image.jpg")
                    print("image created.")
                    w, h = padded_img.size
                    zoom_levels = get_num_zoom_levels(w, h)
                    img.close()
                    tile_image("padded_image.jpg", w, h, zoom_levels, target_dir)

    elif arguments['expandTiff']:
        if arguments['--directory'] is not None:
            target_dir = arguments['--directory']
        if arguments['--tiffFilePath'] is not None:
            print("About to go into the method that expands tiff")
            expand_tiff(arguments['--tiffFilePath'])

    elif arguments['colourAndTileTiff']:
        if arguments['--directory'] is not None:
            target_dir = arguments['--directory']
        if arguments['--newFolderNames'] is not None:
            folder_names = arguments['--newFolderNames']
        expand_colour_tile_tiff(arguments['--tiffFilePath'])
        if arguments['--clean']:
            clean()

    elif arguments['changeHue']:
        try:
            hue = arguments['colour']
        except KeyError:
            hue = 0
        finally:
            if arguments['--directory'] is not None:
                target_dir = arguments['--directory']

            # colorize given image
            count = 1
            for f in arguments['FILE']:
                final_img = colorize_grayscale(f, hue)
                final_img.save(target_dir + 'image' + str(count) + '.jpg')
                count += 1

    elif arguments['--clean']:
        clean()

    elif arguments['--run'] or arguments['-r']:
        app = Flask(__name__)

        @app.route("/")
        def template_test():
            return render_template('index.html', num_layers=num_files, num_slices=num_slices, layers=folder_names,
                                   new_dir=new_dir, original_dir=original_dir)
        app.run(debug=True)

