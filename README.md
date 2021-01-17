# pyLeafletTiler
Collection of code to create tiles of histo images to allow display using the [leaflet JavaScript library](https://leafletjs.com/).

The images can be viewed at different zoom levels, and individual layers can be toggled on/off to show or hide the layer. Markers and annotations can be added (e.g. to indicate ROIs) at different zoom levels, and each layer's brightness, contrast, opacity, and hue are manually adjustable.

## Dependencies
* Imagemagick for the hard part of recursive image tiling
* Leaflet Javascript Library (usually hosted externally)

## Installation
Download the package:

```git clone https://git.sarlab.ca/DrSAR/pyleaflettiler.git```

Install [conda](https://docs.anaconda.com/anaconda/install/) to create and activate the **conda environment** using the following steps in your terminal:
1. Create the environment from the ```environment.yml``` file: ```conda env create -f environment.yml``` 
The first line of the yml file sets the new environment's name. For details see [Creating an environment file manually](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#create-env-file-manually).
2. Activate the new environment: ```conda activate myenv```
3. To verify that the new environment was installed correctly: ```conda env list``` or ```conda info --envs```

## Usage
In your command line / terminal, navigate to this directory using ```cd pyleaflettiler```. The following commands are available:

```
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
```

This is the screen that will be presented when you enter ```python LeafletTiler.py -h``` or ```python LeafletTiler.py --help``` in the command line. Note: ```[optional elements]``` and ```(required elements)```. Commands without brackets must be entered every time.

The program must currently be run by first entering ```python3 -m http.server``` into the command line (in terminal). Note this is a temporary solution (found [here](https://stackoverflow.com/questions/39007243/cannot-open-local-file-chrome-not-allowed-to-load-local-resource)) to the problem with google Chrome that says *'Cannot open local file - Chrome: Not allowed to load local resource'*.

### Workflow:
1. Open the ```config.yaml``` file and adjust the settings according your specific number of slices, and layers in each slice. Note that currently, each slice must be of the same dimension (i.e. same width and height) to avoid stretching.
2. LeafletTiler currently requires layers of the same slice to be aligned and of the same size. This can be checked using the command ```LeafletTiler.py [--test] [FILE] ...```, which will also return the ideal number of zoom levels.
3. Specify the layernames and colours for each layer in the ```config.yaml``` file. This can also be changed using docopt later on (where docopt commands will override the config file). Note that the hue values specified are hues that are applied on top of a sepia filter. Colour examples: 
    - 0 = red
    - 60 = green
    - 90 = light blue
4. Tile the files using the command ```LeafletTiler.py LeafletTiler.py (--directory=<targetdir>) tile [FILE]```. This will save the tiles to a folder named 'tiles' within the specified directory. Alternatively, you can skip the above steps using an ImageJ stack (TIFF File) and pass it into ```LeafletTiler.py [--directory=<targetdir>] colourAndTileTiff (--tiffFilePath=<filepath>) [--newFolderNames=<folders>] [--clean]``` to expand, colour, add padding, and save the tiff in the appropriate directories as specified in the config file (docopt input will overwrite config file if there is input entered).
5. Run ```python3 -m http.server``` in terminal.
6. Create a new terminal window, and enter ```python LeafletTiler.py -r``` or ```python LeafletTiler.py --run```. Then navigate to the given address (e.g. http://127.0.0.1:5000) to view the running application!
