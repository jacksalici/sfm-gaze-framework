
from external.read_write_model import Camera, read_model, qvec2rotmat
from pathlib import Path
import argparse



def printModelImages(model_path) -> None:

    cameras, images, points3D = read_model(Path(model_path) / "sfm", ext=".bin")
    print(images)
    
    for image in images.values():
        print(image.name)
        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process session argument.")
    parser.add_argument('--model_path', '-m', required=True, help="Specify the session ID")

    args = parser.parse_args()
    
    printModelImages(args.model_path)
    

