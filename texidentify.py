import argparse
import os
import shutil
import sys
from pathlib import Path

from pdf2image import convert_from_path

from models import MLPLearningModel

if __name__ == "__main__":
    home = str(Path.home())
    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument("file",
                        help="Path of a PDF file from which handwritten "
                             "equations are to be extracted.", type=str)
    parser.add_argument("--state_dir",
                        help="A directory where model states are stored.",
                        default="home", type=str)
    args = parser.parse_args()

    state_dir = os.path.join(
            home if args.state_dir == "home" else args.state_dir, ".texpredict")
    mlp_model = MLPLearningModel(name="default",
                                 hidden_layer_sizes=(80, ), max_iter=500)
    input_file = args.file

    if not os.path.isfile(input_file):
        raise IOError("Input file %s not found." % input_file)

    if not os.path.isdir(state_dir):
        raise EnvironmentError("Program has never been trained. Please train it"
                               "first.")
    else:
        mlp_model.load(state_dir)
        pages = convert_from_path(input_file, dpi=96,
                                  transparent=False, grayscale=True)
        for index, page in enumerate(pages):
            print("Page %i: %s" % (index, mlp_model.predict_equation(page)))

    print("Program finished.")
