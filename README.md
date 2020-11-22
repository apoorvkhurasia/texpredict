Requirements:
------------
   1. The code has been tested on Linux and macOS and `python3.7` only.
   2. Please install the packages in requirements.txt using the following 
   command:
       ```
       pip3 install -r requirements.txt --user
       ```

Running the app:
----------
   1. A sample training file is under `data`. It is a PDF file with each page 
   containing an 8x8 grid. In each grid there is an image of a symbol. Each
   page contains images for only one symbol. The accompanying labels file in the
   same directory contains the LaTeX syntax for that symbol in order. Number of
   lines in this file must be equal to the number of pages in the PDF file.
   2. To train the model you need files similar to the ones described in step 1.
   Execute the following command to train the model:
        ```
        python3 texlearn.py <training_pdf_file> <labels_txt_file>
        ```
   3. To extract handwriting use the following command:
        ```
        python3 texidentify.py <input_pdf_file>
        ```
      A sample input file is under the `demo` folder. 

Miscellaneous:
-------------   
   1. `model_benchmarking.py` benchmarks various models with various parameters.
   2. `imhelp`, `feature_extraction`, and `models` are modules which
   provide uniform ways to pre-process images, extract features, and train/test
   models.
   3. `plothelp` is a module to plot figures in a standard way.
   
   