
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


def reconstruct(images: Path, outputs: Path, preview = False):
    
    sfm_pairs = outputs / "pairs-sfm.txt"
    sfm_dir = outputs / "sfm"

    features = outputs / "features.h5"
    matches = outputs / "matches.h5"
    loc_pairs = outputs / "pairs-loc.txt"

    #retrieval_conf = extract_features.confs["netvlad"]
    feature_conf = extract_features.confs["superpoint_max"]
    matcher_conf = match_features.confs["superpoint+lightglue"]

    references = [p.relative_to(images).as_posix() for p in images.rglob('*.jpg')]
    print(len(references), "mapping images")
    #plot_images([read_image(images / r) for r in references], dpi=25)

    extract_features.main(
    feature_conf, images, image_list=references, feature_path=features
    )
    pairs_from_exhaustive.main(sfm_pairs, image_list=references)
    match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches);

    model = reconstruction.main(sfm_dir, images, sfm_pairs, features, matches, image_list=references)

    if preview:
        fig = viz_3d.init_figure()
        viz_3d.plot_reconstruction(
            fig, model, color="rgba(255,0,0,0.5)", name="mapping", points_rgb=True
        )
        fig.show()

import multiprocessing
if __name__ == '__main__':
    multiprocessing.freeze_support()  # Only needed for frozen executables

    import tomllib
    config = tomllib.load(open("config.toml", "rb"))
    
    images = Path(config['dataset_path'])
    outputs = Path(config['model_path'])
    
    reconstruct(images, outputs)