

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

import pycolmap
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster


def main():

    images = Path("hloc/dataset")
    query =  "query.jpg"
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
    
    model = reconstruction.main(sfm_dir, images, sfm_pairs, features, matches, image_list=references)



    references_registered = [model.images[i].name for i in model.reg_image_ids()]
    extract_features.main(feature_conf, images, image_list=[query], feature_path=features, overwrite=True)
    pairs_from_exhaustive.main(loc_pairs, image_list=[query], ref_list=references_registered)
    match_features.main(matcher_conf, loc_pairs, features=features, matches=matches, overwrite=True);
    
    fig = viz_3d.init_figure()
    
    camera = pycolmap.infer_camera_from_image(images / query)
    ref_ids = [model.find_image_with_name(n).image_id for n in references_registered]
    conf = {
        'estimation': {'ransac': {'max_error': 12}},
        'refinement': {'refine_focal_length': True, 'refine_extra_params': True},
    }
    localizer = QueryLocalizer(model, conf)
    ret, log = pose_from_cluster(localizer, query, camera, ref_ids, features, matches)

    pose = pycolmap.Image(cam_from_world=ret['cam_from_world'])
    viz_3d.plot_camera_colmap(fig, pose, camera, color='rgba(0,255,0,0.5)', name=query, fill=True)
    # visualize 2D-3D correspodences
    import numpy as np
    inl_3d = np.array([model.points3D[pid].xyz for pid in np.array(log['points3D_ids'])[ret['inliers']]])
    viz_3d.plot_points(fig, inl_3d, color="lime", ps=1, name=query)
    fig.show()
    
import multiprocessing
if __name__ == '__main__':
    multiprocessing.freeze_support()  # Only needed for frozen executables
    main()