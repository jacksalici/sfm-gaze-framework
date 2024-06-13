
from pathlib import Path

from hloc import (
    extract_features,
    match_features,
    reconstruction,
    visualization,
    pairs_from_retrieval,
    pairs_from_exhaustive
)

from hloc.visualization import plot_images, read_image
from hloc.utils import viz_3d


def main():

    images = Path("hloc/dataset")

    outputs = Path("hloc/output/")
    sfm_pairs = outputs / "pairs-sfm.txt"
    sfm_dir = outputs / "sfm"

    features = outputs / "features.h5"
    matches = outputs / "matches.h5"
    loc_pairs = outputs / "pairs-loc.txt"

    retrieval_conf = extract_features.confs["netvlad"]
    feature_conf = extract_features.confs["superpoint_inloc"]
    matcher_conf = match_features.confs["superglue"]


    references = [p.relative_to(images).as_posix() for p in (images).iterdir()]
    print(len(references), "mapping images")
    #plot_images([read_image(images / r) for r in references], dpi=25)


    extract_features.main(
    feature_conf, images, image_list=references, feature_path=features
    )
    pairs_from_exhaustive.main(sfm_pairs, image_list=references)
    match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches);


    model = reconstruction.main(sfm_dir, images, sfm_pairs, features, matches, image_list=references)



    #visualization.visualize_sfm_2d(model, images, color_by="visibility", n=5)


    #visualization.visualize_sfm_2d(model, images, color_by="track_length", n=5)


    #visualization.visualize_sfm_2d(model, images, color_by="depth", n=5)



    fig = viz_3d.init_figure()
    viz_3d.plot_reconstruction(
        fig, model, color="rgba(255,0,0,0.5)", name="mapping", points_rgb=True
    )
    fig.show()

import multiprocessing
if __name__ == '__main__':
    multiprocessing.freeze_support()  # Only needed for frozen executables
    main()