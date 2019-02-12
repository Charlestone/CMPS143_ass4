Everything except the competition part is done in the file "ass4.py". Is can receive the following parameters:
'-tr': File name of the training data. Its default value is "train_examples.tsv".
'-d': File name of the development data.Its default value is "dev_examples.tsv",
'-t': File name of the test data. Its default value is "test_examples.tsv".
'-w': File name of the output. Its default value is None.
'-b': Whether you want to bin the features or not. Its default value is False
'-c': Classifier already trained. Its default value is None.

I have been running the file as: "python ass4.py -w info.txt". Doing so I got almost all the printed information in that file.

restaurant-competition-P2.py has the same parameters described above. To run you can use the following command:

"python restaurant-competition-P2.py -t test_data.tsv -w restaurant-competition-model-p2-predictions.txt"

Also, something I think is important is that I have seen changes in the accuracy values and the confusion matrix throughout the experiments.
Even though they are not big changes, sometimes they can make the best classifier be another that is not the one I picked.