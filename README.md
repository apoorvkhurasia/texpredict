Requirements:
------------
   1. The code has been tested on Linux and macOS only.
   2. Please install the packages in requirements.txt using the following 
   command:
       ```
       pip3 install -r requirements.txt --user
       ```

Running the project:
----------
   1. The training file is under `data`. It is a PDF file with each page 
   containing an 8x8 grid. In each grid there is an image of a symbol. Each
   page contains images for only one symbol. The accompanying labels file in the
   same directory contains the LaTeX syntax for that symbol in order. Number of
   lines in this file must be equal to the number of pages in the PDF file.
   2. `demo.py` trains the model and predicts equations from a given test file.
   An example test file is under the `demo` directory.
   3. `model_benchmarking.py` benchmarks various models with various parameters.
   4. `imhelp`, `feature_extraction`, and `models` are modules which
   provide uniform ways to pre-process images, extract features, and train/test
   models.
   5. `plothelp` is a module to plot figures in a standard way.