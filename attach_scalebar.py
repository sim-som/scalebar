#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import sys
import json
from pathlib import Path
import mrcfile
from skimage import io
from skimage.exposure import equalize_hist
import argparse
# own modules:
metadata_p = Path("../Metadata")    # pylint: disable=import-error
assert metadata_p.exists() and metadata_p.is_dir()
sys.path.insert(1, str(metadata_p))
import metadata as meta_utils # pylint: disable=import-error
# %%
# Function definitions:

def read_img_and_metadata(img_file_p:Path):
    """Read image data (either standard images via skimage.io, or mrcfile via the mrcfile module):

    Args:
        img_file_p (Path):

    Returns:
        _type_: img, px_size
    """

    px_size = 0.0

    if img_file_p.suffix == ".mrc":
        with mrcfile.open(img_file_p) as f:
            img = f.data
            px_size = f.voxel_size.x
    else:
        img = io.imread(img_file_p)
        
    # read the images metadata:
    # whole metadata file:
    whole_metadata_filep = list(img_dir_path.glob(f"metadata of {img_file_p.name[:2]}*.json"))
    assert len(whole_metadata_filep) in [0, 1]
    if len(whole_metadata_filep) == 1:
        whole_metadata_filep = whole_metadata_filep[0]
        with open(whole_metadata_filep, encoding="utf-8") as f:
            whole_metadata = json.load(f)
            file_metadata = meta_utils.get_file_metadata(img_file_p)
        if px_size in [None, 0.0]:
            px_size = meta_utils.get_pixel_size_meter(file_metadata)

    return img, px_size


def attach_scalebar_to_img(img:np.ndarray, px_size:float):

    assert px_size not in [None, 0.0]

    if args.equalize:
        img = equalize_hist(img)

    plt.figure(img_file_p.name)
    plt.imshow(img, cmap="gray")
    plt.axis("off")

    scalebar = ScaleBar(
        px_size,
        location="lower left",
        frameon = args.frameon
    )
    plt.gca().add_artist(scalebar)

    output_img_format = ".png"
    output_p = img_dir_path / Path(f"{img_file_p.stem}_scalebar{output_img_format}")

    plt.savefig(output_p, bbox_inches="tight", pad_inches=0, dpi=args.output_dpi_res)

    if args.verbose:
        plt.show()
    plt.close()
# %%
# Command line parsing:

parser = argparse.ArgumentParser(description=
    "Attach a scalebar to an microscopy image given the pixel size or (implicitly) the metadata file, which has to resign in the same directory"
)
parser.add_argument(
    "img_file",
    type=str,
    help="Filename of the image"
)
parser.add_argument(
    "--px_size",
    type=float,
    default=None,
    help="Overwrite the pixel size of the given image"
)
parser.add_argument("--output_dpi_res", type=int, default=300)
parser.add_argument("-e", "--equalize", action="store_true", help = "Enhance the contrast of the output image")
parser.add_argument("-v", "--verbose", help="Show image with scalebar in separate window. (Great for interactive zooming)",
                    action="store_true")
parser.add_argument("-f", "--frameon", action="store_true", help = "Wheter to draw a box behind the scale bar")
                    

args = parser.parse_args()

img_file_p = Path(args.img_file)
assert img_file_p.exists() and img_file_p.is_file()
img_dir_path = img_file_p.parent

# %%
# main stuff:

img, px_size = read_img_and_metadata(img_file_p)

# overwrite pixel size with potential user input:
if args.px_size:
    px_size = args.px_size

print(f"Image shape: {img.shape} ")
print(f"px size = {px_size} m")

attach_scalebar_to_img(img, px_size)
