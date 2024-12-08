# MiniTorch Module 4

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module4.html

This module requires `fast_ops.py`, `cuda_ops.py`, `scalar.py`, `tensor_functions.py`, `tensor_data.py`, `tensor_ops.py`, `operators.py`, `module.py`, and `autodiff.py` from Module 3.


Additionally you will need to install and download the MNist library.

(On Mac, this may require installing the `wget` command)

```
pip install python-mnist
mnist_get_data.sh
```


* Tests:

```
python run_tests.py
```

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py minitorch/tensor_ops.py minitorch/fast_ops.py minitorch/cuda_ops.py project/parallel_check.py tests/test_tensor_general.py

## Task 4.5 - Model Training Results for SST2 and MNIST

### Sentiment Classification (SST2)

#### Command Used to Run the Model:

```bash
python project/run_sentiment.py
```

#### Model Training Configuration

- Number of training points: 450
- Number of validation points: 100
- Learning rate: 0.01
- Number of epochs: 250

#### Model Training Results

Multiple runs were performed to obtain the best validation accuracy. The final model achieved >70% best validation accuracy on the sentiment classification task.

The full training log can be found in [sentiment.txt](sentiment.txt).

The training log is also shown below, showing the model reaching 74% best validation accuracy:

```console
missing pre-trained embedding for 55 unknown words
Epoch 1, loss 31.260272083233616, train accuracy: 50.22%
Validation accuracy: 53.00%
Best Valid accuracy: 53.00%
Epoch 2, loss 31.141355042186966, train accuracy: 53.56%
Validation accuracy: 56.00%
Best Valid accuracy: 56.00%
Epoch 3, loss 30.996924822823427, train accuracy: 51.56%
Validation accuracy: 57.00%
Best Valid accuracy: 57.00%
Epoch 4, loss 30.818620272651668, train accuracy: 54.67%
Validation accuracy: 54.00%
Best Valid accuracy: 57.00%
Epoch 5, loss 30.538730822041565, train accuracy: 57.78%
Validation accuracy: 51.00%
Best Valid accuracy: 57.00%
Epoch 6, loss 30.243968645056423, train accuracy: 61.56%
Validation accuracy: 55.00%
Best Valid accuracy: 57.00%
Epoch 7, loss 30.051509404984976, train accuracy: 62.22%
Validation accuracy: 60.00%
Best Valid accuracy: 60.00%
Epoch 8, loss 29.57523007125308, train accuracy: 66.22%
Validation accuracy: 59.00%
Best Valid accuracy: 60.00%
Epoch 9, loss 29.234142303381752, train accuracy: 64.67%
Validation accuracy: 53.00%
Best Valid accuracy: 60.00%
Epoch 10, loss 29.146624820452733, train accuracy: 64.44%
Validation accuracy: 68.00%
Best Valid accuracy: 68.00%
Epoch 11, loss 28.690935637596528, train accuracy: 65.78%
Validation accuracy: 61.00%
Best Valid accuracy: 68.00%
Epoch 12, loss 28.140448152894265, train accuracy: 69.11%
Validation accuracy: 62.00%
Best Valid accuracy: 68.00%
Epoch 13, loss 27.66216760106527, train accuracy: 70.89%
Validation accuracy: 61.00%
Best Valid accuracy: 68.00%
Epoch 14, loss 27.035888978772142, train accuracy: 71.56%
Validation accuracy: 69.00%
Best Valid accuracy: 69.00%
Epoch 15, loss 26.77856414084222, train accuracy: 71.56%
Validation accuracy: 71.00%
Best Valid accuracy: 71.00%
Epoch 16, loss 26.008831703369705, train accuracy: 69.78%
Validation accuracy: 71.00%
Best Valid accuracy: 71.00%
Epoch 17, loss 25.29807557819976, train accuracy: 73.78%
Validation accuracy: 70.00%
Best Valid accuracy: 71.00%
Epoch 18, loss 24.59494267972806, train accuracy: 73.78%
Validation accuracy: 69.00%
Best Valid accuracy: 71.00%
Epoch 19, loss 23.66630055745482, train accuracy: 75.11%
Validation accuracy: 62.00%
Best Valid accuracy: 71.00%
Epoch 20, loss 23.632015711113855, train accuracy: 74.22%
Validation accuracy: 70.00%
Best Valid accuracy: 71.00%
Epoch 21, loss 23.22289516521898, train accuracy: 74.22%
Validation accuracy: 72.00%
Best Valid accuracy: 72.00%
Epoch 22, loss 22.287073935500537, train accuracy: 76.44%
Validation accuracy: 72.00%
Best Valid accuracy: 72.00%
Epoch 23, loss 21.940514683154845, train accuracy: 76.22%
Validation accuracy: 71.00%
Best Valid accuracy: 72.00%
Epoch 24, loss 20.816763545046445, train accuracy: 76.89%
Validation accuracy: 71.00%
Best Valid accuracy: 72.00%
Epoch 25, loss 20.71151689891024, train accuracy: 79.78%
Validation accuracy: 73.00%
Best Valid accuracy: 73.00%
Epoch 26, loss 20.180281221853548, train accuracy: 77.11%
Validation accuracy: 73.00%
Best Valid accuracy: 73.00%
Epoch 27, loss 19.52546281738076, train accuracy: 81.56%
Validation accuracy: 74.00%
Best Valid accuracy: 74.00%
Epoch 28, loss 19.110849260752115, train accuracy: 82.22%
Validation accuracy: 67.00%
Best Valid accuracy: 74.00%
Epoch 29, loss 18.042472008117645, train accuracy: 82.67%
Validation accuracy: 74.00%
Best Valid accuracy: 74.00%
Epoch 30, loss 18.508294758702313, train accuracy: 81.11%
Validation accuracy: 71.00%
Best Valid accuracy: 74.00%
Epoch 31, loss 17.056480422438902, train accuracy: 84.00%
Validation accuracy: 68.00%
Best Valid accuracy: 74.00%
Epoch 32, loss 16.92432750474255, train accuracy: 83.56%
Validation accuracy: 69.00%
Best Valid accuracy: 74.00%
Epoch 33, loss 16.768686354671054, train accuracy: 82.67%
Validation accuracy: 71.00%
Best Valid accuracy: 74.00%
Epoch 34, loss 16.520772804864567, train accuracy: 80.00%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 35, loss 15.733854249414579, train accuracy: 83.11%
Validation accuracy: 70.00%
Best Valid accuracy: 74.00%
Epoch 36, loss 15.513621468784281, train accuracy: 81.56%
Validation accuracy: 74.00%
Best Valid accuracy: 74.00%
Epoch 37, loss 15.41843508358301, train accuracy: 85.11%
Validation accuracy: 69.00%
Best Valid accuracy: 74.00%
Epoch 38, loss 15.122401918158383, train accuracy: 83.78%
Validation accuracy: 69.00%
Best Valid accuracy: 74.00%
Epoch 39, loss 14.242036789879823, train accuracy: 82.67%
Validation accuracy: 73.00%
Best Valid accuracy: 74.00%
Epoch 40, loss 13.69415142817124, train accuracy: 82.67%
Validation accuracy: 69.00%
Best Valid accuracy: 74.00%
Epoch 41, loss 13.574763781075509, train accuracy: 85.11%
Validation accuracy: 71.00%
Best Valid accuracy: 74.00%
Epoch 42, loss 13.369252084830476, train accuracy: 86.22%
Validation accuracy: 70.00%
Best Valid accuracy: 74.00%
Epoch 43, loss 13.603602478488675, train accuracy: 86.22%
Validation accuracy: 69.00%
Best Valid accuracy: 74.00%
Epoch 44, loss 12.858379004465815, train accuracy: 85.33%
Validation accuracy: 70.00%
Best Valid accuracy: 74.00%
Epoch 45, loss 12.222113355996854, train accuracy: 86.89%
Validation accuracy: 70.00%
Best Valid accuracy: 74.00%
Epoch 46, loss 14.047285360955838, train accuracy: 84.67%
Validation accuracy: 70.00%
Best Valid accuracy: 74.00%
Epoch 47, loss 12.958137707623976, train accuracy: 84.22%
Validation accuracy: 74.00%
Best Valid accuracy: 74.00%
Epoch 48, loss 12.450736173049414, train accuracy: 85.78%
Validation accuracy: 69.00%
Best Valid accuracy: 74.00%
Epoch 49, loss 12.293601450806733, train accuracy: 86.00%
Validation accuracy: 71.00%
Best Valid accuracy: 74.00%
Epoch 50, loss 11.412413641075611, train accuracy: 87.56%
Validation accuracy: 69.00%
Best Valid accuracy: 74.00%
Epoch 51, loss 11.710639390768893, train accuracy: 86.67%
Validation accuracy: 69.00%
Best Valid accuracy: 74.00%
Epoch 52, loss 12.114491176500922, train accuracy: 84.00%
Validation accuracy: 73.00%
Best Valid accuracy: 74.00%
Epoch 53, loss 10.958041951754824, train accuracy: 88.44%
Validation accuracy: 69.00%
Best Valid accuracy: 74.00%
Epoch 54, loss 12.87366534993347, train accuracy: 83.56%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 55, loss 11.505811303532688, train accuracy: 86.00%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 56, loss 10.976641106877882, train accuracy: 86.22%
Validation accuracy: 69.00%
Best Valid accuracy: 74.00%
Epoch 57, loss 11.280839636688874, train accuracy: 84.00%
Validation accuracy: 73.00%
Best Valid accuracy: 74.00%
Epoch 58, loss 11.027466305217546, train accuracy: 85.33%
Validation accuracy: 68.00%
Best Valid accuracy: 74.00%
Epoch 59, loss 10.789402368259932, train accuracy: 86.00%
Validation accuracy: 68.00%
Best Valid accuracy: 74.00%
Epoch 60, loss 10.493310053583034, train accuracy: 87.56%
Validation accuracy: 68.00%
Best Valid accuracy: 74.00%
Epoch 61, loss 10.215700307295325, train accuracy: 87.78%
Validation accuracy: 67.00%
Best Valid accuracy: 74.00%
Epoch 62, loss 10.17581679407523, train accuracy: 86.89%
Validation accuracy: 73.00%
Best Valid accuracy: 74.00%
Epoch 63, loss 10.84954019597606, train accuracy: 86.44%
Validation accuracy: 69.00%
Best Valid accuracy: 74.00%
Epoch 64, loss 10.668928623805096, train accuracy: 88.00%
Validation accuracy: 69.00%
Best Valid accuracy: 74.00%
Epoch 65, loss 9.758782038475102, train accuracy: 89.78%
Validation accuracy: 67.00%
Best Valid accuracy: 74.00%
Epoch 66, loss 10.205755670761151, train accuracy: 87.11%
Validation accuracy: 67.00%
Best Valid accuracy: 74.00%
Epoch 67, loss 9.69254721506284, train accuracy: 87.11%
Validation accuracy: 68.00%
Best Valid accuracy: 74.00%
Epoch 68, loss 10.289834674048961, train accuracy: 86.89%
Validation accuracy: 67.00%
Best Valid accuracy: 74.00%
Epoch 69, loss 10.300102400707596, train accuracy: 86.67%
Validation accuracy: 68.00%
Best Valid accuracy: 74.00%
Epoch 70, loss 9.815512637645677, train accuracy: 86.44%
Validation accuracy: 69.00%
Best Valid accuracy: 74.00%
Epoch 71, loss 10.38557759005452, train accuracy: 82.22%
Validation accuracy: 71.00%
Best Valid accuracy: 74.00%
Epoch 72, loss 8.344340118368153, train accuracy: 88.89%
Validation accuracy: 69.00%
Best Valid accuracy: 74.00%
Epoch 73, loss 9.750188518752203, train accuracy: 86.67%
Validation accuracy: 69.00%
Best Valid accuracy: 74.00%
Epoch 74, loss 9.200202440172589, train accuracy: 88.44%
Validation accuracy: 69.00%
Best Valid accuracy: 74.00%
Epoch 75, loss 10.765560704416144, train accuracy: 85.78%
Validation accuracy: 69.00%
Best Valid accuracy: 74.00%
Epoch 76, loss 9.541360362577882, train accuracy: 88.00%
Validation accuracy: 69.00%
Best Valid accuracy: 74.00%
Epoch 77, loss 9.59559695622629, train accuracy: 86.89%
Validation accuracy: 68.00%
Best Valid accuracy: 74.00%
Epoch 78, loss 10.34892852800455, train accuracy: 84.89%
Validation accuracy: 69.00%
Best Valid accuracy: 74.00%
Epoch 79, loss 10.207710850275967, train accuracy: 87.11%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 80, loss 8.570043480012119, train accuracy: 87.11%
Validation accuracy: 70.00%
Best Valid accuracy: 74.00%
Epoch 81, loss 9.314508943950809, train accuracy: 86.22%
Validation accuracy: 70.00%
Best Valid accuracy: 74.00%
Epoch 82, loss 9.770214963682895, train accuracy: 86.44%
Validation accuracy: 71.00%
Best Valid accuracy: 74.00%
Epoch 83, loss 8.708220738068139, train accuracy: 88.67%
Validation accuracy: 69.00%
Best Valid accuracy: 74.00%
Epoch 84, loss 9.438734268745524, train accuracy: 87.56%
Validation accuracy: 70.00%
Best Valid accuracy: 74.00%
Epoch 85, loss 9.855270513925365, train accuracy: 84.89%
Validation accuracy: 70.00%
Best Valid accuracy: 74.00%
Epoch 86, loss 9.677412090725705, train accuracy: 87.11%
Validation accuracy: 70.00%
Best Valid accuracy: 74.00%
Epoch 87, loss 8.328774893965843, train accuracy: 88.44%
Validation accuracy: 70.00%
Best Valid accuracy: 74.00%
Epoch 88, loss 9.098674162334781, train accuracy: 87.33%
Validation accuracy: 69.00%
Best Valid accuracy: 74.00%
Epoch 89, loss 9.663497484519594, train accuracy: 88.00%
Validation accuracy: 69.00%
Best Valid accuracy: 74.00%
Epoch 90, loss 8.741440560515631, train accuracy: 87.78%
Validation accuracy: 70.00%
Best Valid accuracy: 74.00%
Epoch 91, loss 8.20563312564053, train accuracy: 89.11%
Validation accuracy: 70.00%
Best Valid accuracy: 74.00%
Epoch 92, loss 8.540681493740019, train accuracy: 88.89%
Validation accuracy: 69.00%
Best Valid accuracy: 74.00%
Epoch 93, loss 9.478786696575726, train accuracy: 87.56%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 94, loss 9.542630243520371, train accuracy: 85.56%
Validation accuracy: 71.00%
Best Valid accuracy: 74.00%
Epoch 95, loss 8.749883083240519, train accuracy: 88.00%
Validation accuracy: 68.00%
Best Valid accuracy: 74.00%
Epoch 96, loss 9.875875736833658, train accuracy: 88.00%
Validation accuracy: 69.00%
Best Valid accuracy: 74.00%
Epoch 97, loss 8.793469422322305, train accuracy: 87.11%
Validation accuracy: 70.00%
Best Valid accuracy: 74.00%
Epoch 98, loss 8.175135480860844, train accuracy: 85.78%
Validation accuracy: 68.00%
Best Valid accuracy: 74.00%
Epoch 99, loss 9.48392419905917, train accuracy: 85.78%
Validation accuracy: 71.00%
Best Valid accuracy: 74.00%
Epoch 100, loss 8.51964016061218, train accuracy: 87.56%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 101, loss 8.726802565915088, train accuracy: 86.44%
Validation accuracy: 73.00%
Best Valid accuracy: 74.00%
Epoch 102, loss 8.822678689735586, train accuracy: 87.78%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 103, loss 10.45888030183302, train accuracy: 84.44%
Validation accuracy: 69.00%
Best Valid accuracy: 74.00%
Epoch 104, loss 9.055250600318924, train accuracy: 86.89%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 105, loss 8.427523892837101, train accuracy: 89.11%
Validation accuracy: 70.00%
Best Valid accuracy: 74.00%
Epoch 106, loss 9.135659166267633, train accuracy: 86.44%
Validation accuracy: 71.00%
Best Valid accuracy: 74.00%
Epoch 107, loss 8.14163757149296, train accuracy: 86.67%
Validation accuracy: 71.00%
Best Valid accuracy: 74.00%
Epoch 108, loss 9.355110122369332, train accuracy: 84.00%
Validation accuracy: 71.00%
Best Valid accuracy: 74.00%
Epoch 109, loss 9.169598764805823, train accuracy: 88.44%
Validation accuracy: 71.00%
Best Valid accuracy: 74.00%
Epoch 110, loss 8.644700592358987, train accuracy: 87.11%
Validation accuracy: 71.00%
Best Valid accuracy: 74.00%
Epoch 111, loss 8.073599076178864, train accuracy: 87.56%
Validation accuracy: 70.00%
Best Valid accuracy: 74.00%
Epoch 112, loss 8.360794259159038, train accuracy: 87.78%
Validation accuracy: 70.00%
Best Valid accuracy: 74.00%
Epoch 113, loss 8.727221355357816, train accuracy: 86.67%
Validation accuracy: 71.00%
Best Valid accuracy: 74.00%
Epoch 114, loss 8.995593567256138, train accuracy: 86.00%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 115, loss 7.552529969469354, train accuracy: 89.56%
Validation accuracy: 69.00%
Best Valid accuracy: 74.00%
Epoch 116, loss 8.7228487764355, train accuracy: 90.22%
Validation accuracy: 70.00%
Best Valid accuracy: 74.00%
Epoch 117, loss 8.511129146413083, train accuracy: 87.11%
Validation accuracy: 70.00%
Best Valid accuracy: 74.00%
Epoch 118, loss 9.42183765902901, train accuracy: 85.78%
Validation accuracy: 71.00%
Best Valid accuracy: 74.00%
Epoch 119, loss 8.185031343694291, train accuracy: 86.22%
Validation accuracy: 71.00%
Best Valid accuracy: 74.00%
Epoch 120, loss 7.610443475287271, train accuracy: 88.89%
Validation accuracy: 68.00%
Best Valid accuracy: 74.00%
Epoch 121, loss 7.7071353638631175, train accuracy: 85.78%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 122, loss 9.103164724382935, train accuracy: 86.22%
Validation accuracy: 71.00%
Best Valid accuracy: 74.00%
Epoch 123, loss 7.871124103224756, train accuracy: 89.56%
Validation accuracy: 70.00%
Best Valid accuracy: 74.00%
Epoch 124, loss 9.731187231877387, train accuracy: 82.22%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 125, loss 9.138884950425327, train accuracy: 86.89%
Validation accuracy: 70.00%
Best Valid accuracy: 74.00%
Epoch 126, loss 7.810205095601123, train accuracy: 88.44%
Validation accuracy: 70.00%
Best Valid accuracy: 74.00%
Epoch 127, loss 8.96379192338935, train accuracy: 86.89%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 128, loss 9.464393261909438, train accuracy: 84.67%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 129, loss 7.67607577983136, train accuracy: 86.89%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 130, loss 8.291391400741022, train accuracy: 88.00%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 131, loss 9.392703576020423, train accuracy: 86.22%
Validation accuracy: 70.00%
Best Valid accuracy: 74.00%
Epoch 132, loss 7.832555497724109, train accuracy: 86.67%
Validation accuracy: 70.00%
Best Valid accuracy: 74.00%
Epoch 133, loss 8.974784591701058, train accuracy: 85.56%
Validation accuracy: 70.00%
Best Valid accuracy: 74.00%
Epoch 134, loss 8.11959911823327, train accuracy: 87.56%
Validation accuracy: 68.00%
Best Valid accuracy: 74.00%
Epoch 135, loss 9.719542194026827, train accuracy: 84.67%
Validation accuracy: 68.00%
Best Valid accuracy: 74.00%
Epoch 136, loss 9.054635167293025, train accuracy: 85.56%
Validation accuracy: 70.00%
Best Valid accuracy: 74.00%
Epoch 137, loss 8.472046570878945, train accuracy: 86.22%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 138, loss 7.858387057316264, train accuracy: 88.22%
Validation accuracy: 68.00%
Best Valid accuracy: 74.00%
Epoch 139, loss 8.882781396123555, train accuracy: 84.67%
Validation accuracy: 71.00%
Best Valid accuracy: 74.00%
Epoch 140, loss 8.22823945363362, train accuracy: 88.44%
Validation accuracy: 70.00%
Best Valid accuracy: 74.00%
Epoch 141, loss 8.097391874824497, train accuracy: 87.11%
Validation accuracy: 71.00%
Best Valid accuracy: 74.00%
Epoch 142, loss 7.856633070332042, train accuracy: 85.33%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 143, loss 8.722464702500853, train accuracy: 86.22%
Validation accuracy: 69.00%
Best Valid accuracy: 74.00%
Epoch 144, loss 8.180992003637005, train accuracy: 87.56%
Validation accuracy: 70.00%
Best Valid accuracy: 74.00%
Epoch 145, loss 7.336613305014945, train accuracy: 90.00%
Validation accuracy: 70.00%
Best Valid accuracy: 74.00%
Epoch 146, loss 7.934906908733712, train accuracy: 88.89%
Validation accuracy: 70.00%
Best Valid accuracy: 74.00%
Epoch 147, loss 7.602399528117027, train accuracy: 90.44%
Validation accuracy: 70.00%
Best Valid accuracy: 74.00%
Epoch 148, loss 8.198396520066746, train accuracy: 85.11%
Validation accuracy: 71.00%
Best Valid accuracy: 74.00%
Epoch 149, loss 8.122195508745433, train accuracy: 88.67%
Validation accuracy: 71.00%
Best Valid accuracy: 74.00%
Epoch 150, loss 8.827987655757612, train accuracy: 86.00%
Validation accuracy: 71.00%
Best Valid accuracy: 74.00%
Epoch 151, loss 9.15081992619229, train accuracy: 86.67%
Validation accuracy: 71.00%
Best Valid accuracy: 74.00%
Epoch 152, loss 7.939498121943015, train accuracy: 88.89%
Validation accuracy: 70.00%
Best Valid accuracy: 74.00%
Epoch 153, loss 8.66827848135066, train accuracy: 86.44%
Validation accuracy: 71.00%
Best Valid accuracy: 74.00%
Epoch 154, loss 8.882214747680484, train accuracy: 86.22%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 155, loss 8.33130732382591, train accuracy: 85.78%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 156, loss 7.025658165527218, train accuracy: 89.78%
Validation accuracy: 71.00%
Best Valid accuracy: 74.00%
Epoch 157, loss 8.060968559324339, train accuracy: 85.33%
Validation accuracy: 71.00%
Best Valid accuracy: 74.00%
Epoch 158, loss 7.897494597606201, train accuracy: 87.56%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 159, loss 8.027298041590218, train accuracy: 88.44%
Validation accuracy: 71.00%
Best Valid accuracy: 74.00%
Epoch 160, loss 7.495879147795516, train accuracy: 87.78%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 161, loss 7.557897291264424, train accuracy: 88.44%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 162, loss 7.600635097086528, train accuracy: 87.56%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 163, loss 7.563042704959456, train accuracy: 88.00%
Validation accuracy: 71.00%
Best Valid accuracy: 74.00%
Epoch 164, loss 8.689816028502818, train accuracy: 86.22%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 165, loss 8.346470592176908, train accuracy: 86.00%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 166, loss 8.000992044320595, train accuracy: 89.78%
Validation accuracy: 70.00%
Best Valid accuracy: 74.00%
Epoch 167, loss 8.714734921745874, train accuracy: 86.00%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 168, loss 7.431504034423948, train accuracy: 87.33%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 169, loss 8.77023806377037, train accuracy: 85.78%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 170, loss 8.104120286551337, train accuracy: 88.44%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 171, loss 8.588603111231873, train accuracy: 87.78%
Validation accuracy: 71.00%
Best Valid accuracy: 74.00%
Epoch 172, loss 9.805372216366553, train accuracy: 83.78%
Validation accuracy: 73.00%
Best Valid accuracy: 74.00%
Epoch 173, loss 7.49998220026758, train accuracy: 87.78%
Validation accuracy: 70.00%
Best Valid accuracy: 74.00%
Epoch 174, loss 9.387536400482222, train accuracy: 86.22%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 175, loss 6.8669440505615, train accuracy: 87.33%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 176, loss 8.971625895704822, train accuracy: 85.78%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 177, loss 8.098039437683942, train accuracy: 86.67%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 178, loss 8.175562818508912, train accuracy: 85.78%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 179, loss 9.298607657685972, train accuracy: 85.33%
Validation accuracy: 71.00%
Best Valid accuracy: 74.00%
Epoch 180, loss 8.07502352099695, train accuracy: 87.33%
Validation accuracy: 73.00%
Best Valid accuracy: 74.00%
Epoch 181, loss 7.763400026514633, train accuracy: 86.67%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 182, loss 8.385987128225137, train accuracy: 85.33%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 183, loss 7.901336053729714, train accuracy: 88.44%
Validation accuracy: 70.00%
Best Valid accuracy: 74.00%
Epoch 184, loss 7.721920679179595, train accuracy: 89.56%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 185, loss 8.042155810762402, train accuracy: 85.78%
Validation accuracy: 71.00%
Best Valid accuracy: 74.00%
Epoch 186, loss 8.525726616195357, train accuracy: 84.89%
Validation accuracy: 73.00%
Best Valid accuracy: 74.00%
Epoch 187, loss 9.139101766956067, train accuracy: 85.33%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 188, loss 7.930946458968892, train accuracy: 88.00%
Validation accuracy: 71.00%
Best Valid accuracy: 74.00%
Epoch 189, loss 8.186377647481025, train accuracy: 87.33%
Validation accuracy: 71.00%
Best Valid accuracy: 74.00%
Epoch 190, loss 7.898124509174589, train accuracy: 85.78%
Validation accuracy: 73.00%
Best Valid accuracy: 74.00%
Epoch 191, loss 8.176344390245367, train accuracy: 87.11%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 192, loss 8.570434162572441, train accuracy: 85.56%
Validation accuracy: 71.00%
Best Valid accuracy: 74.00%
Epoch 193, loss 7.892068702784787, train accuracy: 85.56%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 194, loss 8.033770356982771, train accuracy: 87.78%
Validation accuracy: 71.00%
Best Valid accuracy: 74.00%
Epoch 195, loss 8.532373473368045, train accuracy: 87.33%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 196, loss 7.78498365956711, train accuracy: 88.22%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 197, loss 8.457873004728054, train accuracy: 86.00%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 198, loss 8.254233206960707, train accuracy: 87.11%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 199, loss 8.346453240348655, train accuracy: 85.78%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 200, loss 9.099753136068275, train accuracy: 86.44%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 201, loss 8.354264907634935, train accuracy: 87.78%
Validation accuracy: 71.00%
Best Valid accuracy: 74.00%
Epoch 202, loss 8.157925669678452, train accuracy: 86.44%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 203, loss 8.768641499308552, train accuracy: 86.00%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 204, loss 7.5687956841597295, train accuracy: 88.22%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 205, loss 6.8692574756157425, train accuracy: 89.78%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 206, loss 7.931665617219712, train accuracy: 88.00%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 207, loss 8.16388653620624, train accuracy: 86.22%
Validation accuracy: 71.00%
Best Valid accuracy: 74.00%
Epoch 208, loss 8.317989638630051, train accuracy: 88.00%
Validation accuracy: 71.00%
Best Valid accuracy: 74.00%
Epoch 209, loss 8.62002006191861, train accuracy: 86.89%
Validation accuracy: 71.00%
Best Valid accuracy: 74.00%
Epoch 210, loss 9.35249831388172, train accuracy: 84.67%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 211, loss 8.12352580136892, train accuracy: 87.33%
Validation accuracy: 71.00%
Best Valid accuracy: 74.00%
Epoch 212, loss 8.432869589859166, train accuracy: 88.67%
Validation accuracy: 71.00%
Best Valid accuracy: 74.00%
Epoch 213, loss 7.773309191846218, train accuracy: 86.00%
Validation accuracy: 73.00%
Best Valid accuracy: 74.00%
Epoch 214, loss 8.319266108287831, train accuracy: 88.00%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 215, loss 8.95163057850833, train accuracy: 86.67%
Validation accuracy: 71.00%
Best Valid accuracy: 74.00%
Epoch 216, loss 6.982791630170979, train accuracy: 88.00%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 217, loss 8.574524368050858, train accuracy: 86.89%
Validation accuracy: 71.00%
Best Valid accuracy: 74.00%
Epoch 218, loss 7.101543892644403, train accuracy: 88.44%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 219, loss 7.423912468047777, train accuracy: 88.89%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 220, loss 8.729606056143039, train accuracy: 86.22%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 221, loss 7.967854345337939, train accuracy: 88.44%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 222, loss 7.730903763915215, train accuracy: 86.89%
Validation accuracy: 71.00%
Best Valid accuracy: 74.00%
Epoch 223, loss 7.793544923649549, train accuracy: 89.78%
Validation accuracy: 71.00%
Best Valid accuracy: 74.00%
Epoch 224, loss 8.85917673433996, train accuracy: 85.78%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 225, loss 7.810827658408127, train accuracy: 86.89%
Validation accuracy: 73.00%
Best Valid accuracy: 74.00%
Epoch 226, loss 7.46623857938203, train accuracy: 86.00%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 227, loss 7.551992837973532, train accuracy: 89.78%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 228, loss 7.1908403544697235, train accuracy: 88.89%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 229, loss 7.934217016312446, train accuracy: 87.33%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 230, loss 8.216852229507339, train accuracy: 86.44%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 231, loss 8.071970602435389, train accuracy: 86.89%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 232, loss 8.665041320131305, train accuracy: 84.44%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 233, loss 8.043901057615107, train accuracy: 87.33%
Validation accuracy: 71.00%
Best Valid accuracy: 74.00%
Epoch 234, loss 9.570874842324228, train accuracy: 85.11%
Validation accuracy: 71.00%
Best Valid accuracy: 74.00%
Epoch 235, loss 8.21203157737752, train accuracy: 87.56%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 236, loss 8.213495289584422, train accuracy: 86.89%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 237, loss 7.8276749773243814, train accuracy: 87.78%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 238, loss 8.458029421494636, train accuracy: 83.78%
Validation accuracy: 73.00%
Best Valid accuracy: 74.00%
Epoch 239, loss 8.070483736664238, train accuracy: 88.00%
Validation accuracy: 71.00%
Best Valid accuracy: 74.00%
Epoch 240, loss 8.257698810323848, train accuracy: 85.11%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 241, loss 8.819788785964, train accuracy: 86.00%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 242, loss 7.535919735475292, train accuracy: 88.00%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 243, loss 9.15887997655674, train accuracy: 85.56%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 244, loss 7.924855986993845, train accuracy: 88.44%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 245, loss 6.885462426916719, train accuracy: 89.11%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 246, loss 8.201071757933986, train accuracy: 88.22%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 247, loss 8.581961531388504, train accuracy: 86.44%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 248, loss 8.724963488021526, train accuracy: 84.44%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 249, loss 8.539017286459336, train accuracy: 85.78%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
Epoch 250, loss 8.466766387128647, train accuracy: 87.11%
Validation accuracy: 72.00%
Best Valid accuracy: 74.00%
```

### Digit Classification (MNIST)

#### Command Used to Run the Model:

```bash
python project/run_mnist_multiclass.py
```

#### Model Training Configuration

- Number of training points: 5000
- Number of validation points: 500
- Learning rate: 0.01
- Number of epochs: 25

#### Model Training Results

Multiple runs were performed to obtain the best validation accuracy. The final model achieved majority 16/16 classification accuracy on the MNIST validation set.

The full training log can be found in [mnist.txt](mnist.txt).

The training log is also shown below, showing the model reaching majority 16/16 validation classification accuracy when the epoch is 16 (There is not need to train any longer as continue training may result in overfitting and early termination is appropriate in this case):

```console
Epoch 1 loss 2.3077798541026606 valid acc 2/16
Epoch 1 loss 11.465867158181752 valid acc 2/16
Epoch 1 loss 11.461627497727573 valid acc 2/16
Epoch 1 loss 11.302066744339074 valid acc 3/16
Epoch 1 loss 10.97501968995342 valid acc 6/16
Epoch 1 loss 10.668748690743886 valid acc 5/16
Epoch 1 loss 9.208279762130273 valid acc 9/16
Epoch 1 loss 8.854357489648695 valid acc 8/16
Epoch 1 loss 10.169762611980897 valid acc 10/16
Epoch 1 loss 8.897657427180517 valid acc 9/16
Epoch 1 loss 8.076754768689462 valid acc 8/16
Epoch 1 loss 8.55774093407289 valid acc 10/16
Epoch 1 loss 7.467443403425016 valid acc 13/16
Epoch 1 loss 6.92730100849383 valid acc 12/16
Epoch 1 loss 6.964216349726793 valid acc 13/16
Epoch 1 loss 6.81142063585056 valid acc 7/16
Epoch 1 loss 7.5689665769465755 valid acc 10/16
Epoch 1 loss 6.550201604160344 valid acc 10/16
Epoch 1 loss 5.913393765629233 valid acc 14/16
Epoch 1 loss 4.95533031474335 valid acc 14/16
Epoch 1 loss 4.865376424466178 valid acc 12/16
Epoch 1 loss 4.319958603867515 valid acc 12/16
Epoch 1 loss 2.1912604791065107 valid acc 12/16
Epoch 1 loss 4.043775871884099 valid acc 15/16
Epoch 1 loss 4.021497056241549 valid acc 11/16
Epoch 1 loss 3.8841190346341268 valid acc 9/16
Epoch 1 loss 5.917745232119165 valid acc 14/16
Epoch 1 loss 3.0099733140549803 valid acc 13/16
Epoch 1 loss 4.667959212777786 valid acc 13/16
Epoch 1 loss 3.3712466176318046 valid acc 10/16
Epoch 1 loss 4.568897576348359 valid acc 13/16
Epoch 1 loss 2.811912263197977 valid acc 12/16
Epoch 1 loss 3.724038198495855 valid acc 14/16
Epoch 1 loss 3.1166672913632 valid acc 15/16
Epoch 1 loss 5.383438640440718 valid acc 15/16
Epoch 1 loss 3.6683886002265 valid acc 14/16
Epoch 1 loss 2.869044769584521 valid acc 14/16
Epoch 1 loss 3.3834408003550367 valid acc 14/16
Epoch 1 loss 3.286344374086722 valid acc 14/16
Epoch 1 loss 2.8004764128905033 valid acc 13/16
Epoch 1 loss 1.8534627016106078 valid acc 14/16
Epoch 1 loss 2.952626786753548 valid acc 14/16
Epoch 1 loss 2.7294566494306114 valid acc 15/16
Epoch 1 loss 2.4131525837131313 valid acc 13/16
Epoch 1 loss 3.0230940689229744 valid acc 13/16
Epoch 1 loss 1.509365003375176 valid acc 16/16
Epoch 1 loss 2.6461024638096333 valid acc 15/16
Epoch 1 loss 2.6757953151893985 valid acc 16/16
Epoch 1 loss 2.2182161355556476 valid acc 14/16
Epoch 1 loss 2.7023822647414253 valid acc 15/16
Epoch 1 loss 2.369634233026005 valid acc 15/16
Epoch 1 loss 2.485718411168529 valid acc 15/16
Epoch 1 loss 2.5792695956328746 valid acc 15/16
Epoch 1 loss 1.7018420260511316 valid acc 15/16
Epoch 1 loss 3.4415677304361134 valid acc 15/16
Epoch 1 loss 2.095210461804668 valid acc 14/16
Epoch 1 loss 2.523302709357538 valid acc 16/16
Epoch 1 loss 2.2514551424535165 valid acc 16/16
Epoch 1 loss 2.7016923027418005 valid acc 15/16
Epoch 1 loss 2.6132173966121917 valid acc 16/16
Epoch 1 loss 2.062825046988001 valid acc 15/16
Epoch 1 loss 2.242014364418709 valid acc 16/16
Epoch 1 loss 2.7241048592941772 valid acc 15/16
Epoch 2 loss 0.07735407993531657 valid acc 16/16
Epoch 2 loss 1.717274265184833 valid acc 15/16
Epoch 2 loss 2.6549650914540517 valid acc 15/16
Epoch 2 loss 2.135527002868762 valid acc 15/16
Epoch 2 loss 1.1276217579245136 valid acc 15/16
Epoch 2 loss 1.5857747444908334 valid acc 15/16
Epoch 2 loss 2.116686849109519 valid acc 14/16
Epoch 2 loss 2.3244597166043395 valid acc 15/16
Epoch 2 loss 2.0788245170024267 valid acc 15/16
Epoch 2 loss 2.2945006536547305 valid acc 15/16
Epoch 2 loss 2.020185964522041 valid acc 15/16
Epoch 2 loss 3.0142736695590564 valid acc 16/16
Epoch 2 loss 2.4423332061923526 valid acc 16/16
Epoch 2 loss 3.3209356541767567 valid acc 16/16
Epoch 2 loss 2.2758499875121236 valid acc 16/16
Epoch 2 loss 1.8866145428340424 valid acc 16/16
Epoch 2 loss 3.930251388073529 valid acc 16/16
Epoch 2 loss 2.384042042088853 valid acc 16/16
Epoch 2 loss 1.5199355446154768 valid acc 16/16
Epoch 2 loss 1.4211015875302584 valid acc 16/16
Epoch 2 loss 1.3794623950062974 valid acc 16/16
Epoch 2 loss 1.7057256471197042 valid acc 16/16
Epoch 2 loss 0.68740973337893 valid acc 16/16
Epoch 2 loss 1.8512660625985844 valid acc 16/16
Epoch 2 loss 1.5459014817375285 valid acc 15/16
Epoch 2 loss 1.5532776667242083 valid acc 15/16
Epoch 2 loss 2.1279348913780898 valid acc 15/16
Epoch 2 loss 0.7827867951552493 valid acc 16/16
Epoch 2 loss 0.7267341967978049 valid acc 16/16
Epoch 2 loss 1.0133125834987702 valid acc 16/16
Epoch 2 loss 1.6736839655671378 valid acc 16/16
Epoch 2 loss 1.465896232184629 valid acc 15/16
Epoch 2 loss 1.1658156024557889 valid acc 14/16
Epoch 2 loss 1.430261078086859 valid acc 15/16
Epoch 2 loss 3.5772123941082095 valid acc 16/16
Epoch 2 loss 1.2910034277250495 valid acc 16/16
Epoch 2 loss 1.6382311814055488 valid acc 15/16
Epoch 2 loss 1.4922708655165313 valid acc 15/16
Epoch 2 loss 1.8694866790024196 valid acc 15/16
Epoch 2 loss 1.7751535784267891 valid acc 14/16
Epoch 2 loss 1.192013481496581 valid acc 15/16
Epoch 2 loss 2.109963876773078 valid acc 16/16
Epoch 2 loss 1.4575650631957329 valid acc 16/16
Epoch 2 loss 1.0621775201228585 valid acc 15/16
Epoch 2 loss 1.8670383314880743 valid acc 16/16
Epoch 2 loss 0.48035543906079237 valid acc 16/16
Epoch 2 loss 1.8022672136126607 valid acc 16/16
Epoch 2 loss 2.405085440006493 valid acc 15/16
Epoch 2 loss 1.0303475463686245 valid acc 15/16
Epoch 2 loss 1.4963149648725778 valid acc 16/16
Epoch 2 loss 1.2090817508339844 valid acc 16/16
Epoch 2 loss 1.239657615184647 valid acc 16/16
Epoch 2 loss 1.8837670188924647 valid acc 14/16
Epoch 2 loss 0.9148119291182709 valid acc 16/16
Epoch 2 loss 1.9211024425994716 valid acc 15/16
Epoch 2 loss 1.0332855371808756 valid acc 16/16
Epoch 2 loss 0.8932918795912371 valid acc 16/16
Epoch 2 loss 1.324093905626673 valid acc 16/16
Epoch 2 loss 1.779989240160941 valid acc 16/16
Epoch 2 loss 1.2779951182895586 valid acc 16/16
Epoch 2 loss 1.4445507678944702 valid acc 16/16
Epoch 2 loss 1.4445332129372568 valid acc 16/16
Epoch 2 loss 2.487015292536179 valid acc 16/16
Epoch 3 loss 0.06141827520573401 valid acc 16/16
Epoch 3 loss 1.3062497836381313 valid acc 16/16
Epoch 3 loss 2.0664372836099663 valid acc 15/16
Epoch 3 loss 1.7157219444269098 valid acc 16/16
Epoch 3 loss 0.9134525660328175 valid acc 15/16
Epoch 3 loss 1.1446025234691184 valid acc 16/16
Epoch 3 loss 1.6417220964577512 valid acc 16/16
Epoch 3 loss 1.467433733641591 valid acc 15/16
Epoch 3 loss 1.0520749001684186 valid acc 15/16
Epoch 3 loss 1.2897746787787405 valid acc 16/16
Epoch 3 loss 0.8428915194843358 valid acc 16/16
Epoch 3 loss 2.192209164079375 valid acc 16/16
Epoch 3 loss 1.3889275358298052 valid acc 16/16
Epoch 3 loss 2.033751393286593 valid acc 15/16
Epoch 3 loss 2.265851442670805 valid acc 16/16
Epoch 3 loss 0.8859745595908239 valid acc 16/16
Epoch 3 loss 2.1597568602252175 valid acc 15/16
Epoch 3 loss 2.1389705093432845 valid acc 16/16
Epoch 3 loss 2.39422733769883 valid acc 15/16
Epoch 3 loss 0.9103272118567945 valid acc 15/16
Epoch 3 loss 1.093412138300556 valid acc 14/16
Epoch 3 loss 1.008559256605877 valid acc 16/16
Epoch 3 loss 0.4417068222940976 valid acc 15/16
Epoch 3 loss 1.1101304970000834 valid acc 16/16
Epoch 3 loss 0.8174025334306048 valid acc 15/16
Epoch 3 loss 0.8517647479288688 valid acc 15/16
Epoch 3 loss 0.8357035197861586 valid acc 16/16
Epoch 3 loss 1.2647482759025634 valid acc 16/16
Epoch 3 loss 0.6340438556124433 valid acc 16/16
Epoch 3 loss 0.4690654165540108 valid acc 16/16
Epoch 3 loss 2.3081050283473004 valid acc 15/16
Epoch 3 loss 1.371187836568523 valid acc 15/16
Epoch 3 loss 0.6013743280797103 valid acc 16/16
Epoch 3 loss 0.882088518346211 valid acc 15/16
Epoch 3 loss 2.8006524906026677 valid acc 16/16
Epoch 3 loss 0.8223886524014598 valid acc 16/16
Epoch 3 loss 0.7310278103848031 valid acc 15/16
Epoch 3 loss 1.2626574513562394 valid acc 16/16
Epoch 3 loss 0.7890609157627986 valid acc 16/16
Epoch 3 loss 0.8685866089433015 valid acc 16/16
Epoch 3 loss 0.5288203854408077 valid acc 15/16
Epoch 3 loss 1.7350406721007634 valid acc 16/16
Epoch 3 loss 1.277889416191945 valid acc 16/16
Epoch 3 loss 0.5613080539080126 valid acc 16/16
Epoch 3 loss 1.0800041257173747 valid acc 16/16
Epoch 3 loss 0.527533906389227 valid acc 16/16
Epoch 3 loss 0.8337378556322717 valid acc 16/16
Epoch 3 loss 2.0518354764507802 valid acc 15/16
Epoch 3 loss 0.42779951018361 valid acc 16/16
Epoch 3 loss 0.8837566138973123 valid acc 16/16
Epoch 3 loss 0.46740365741233736 valid acc 15/16
Epoch 3 loss 0.9592016428312601 valid acc 16/16
Epoch 3 loss 1.2277189820034375 valid acc 15/16
Epoch 3 loss 0.7071404359175222 valid acc 16/16
Epoch 3 loss 1.4087121649651397 valid acc 16/16
Epoch 3 loss 0.5873333710677626 valid acc 16/16
Epoch 3 loss 0.8340731718415937 valid acc 15/16
Epoch 3 loss 1.2446707914534008 valid acc 16/16
Epoch 3 loss 1.768362648652954 valid acc 16/16
Epoch 3 loss 1.0181557128808247 valid acc 16/16
Epoch 3 loss 0.9023739767521792 valid acc 16/16
Epoch 3 loss 0.5549269510839834 valid acc 15/16
Epoch 3 loss 1.1817843712231437 valid acc 16/16
Epoch 4 loss 0.06443398868864098 valid acc 16/16
Epoch 4 loss 1.192175624359714 valid acc 16/16
Epoch 4 loss 1.3429389186566172 valid acc 16/16
Epoch 4 loss 0.6365766745585151 valid acc 16/16
Epoch 4 loss 0.5352560470471923 valid acc 16/16
Epoch 4 loss 0.5843954469043119 valid acc 15/16
Epoch 4 loss 1.550874769701474 valid acc 16/16
Epoch 4 loss 1.1964027295456416 valid acc 16/16
Epoch 4 loss 0.5349789335756642 valid acc 16/16
Epoch 4 loss 0.476943426774323 valid acc 16/16
Epoch 4 loss 0.5552705641385881 valid acc 16/16
Epoch 4 loss 1.6138971755681157 valid acc 15/16
Epoch 4 loss 1.5774137677936104 valid acc 16/16
Epoch 4 loss 1.4065458378355964 valid acc 16/16
Epoch 4 loss 1.5992704991895867 valid acc 16/16
Epoch 4 loss 0.665178119838836 valid acc 16/16
Epoch 4 loss 1.729728758621179 valid acc 16/16
Epoch 4 loss 1.3702490644417074 valid acc 16/16
Epoch 4 loss 0.94257701600456 valid acc 16/16
Epoch 4 loss 0.4876524813534547 valid acc 16/16
Epoch 4 loss 1.4894410195631451 valid acc 15/16
Epoch 4 loss 1.4589707391207758 valid acc 16/16
Epoch 4 loss 0.4162960491885869 valid acc 16/16
Epoch 4 loss 1.2382741149037306 valid acc 16/16
Epoch 4 loss 0.7659208390486609 valid acc 15/16
Epoch 4 loss 1.1025993263990734 valid acc 16/16
Epoch 4 loss 0.5923985469981772 valid acc 16/16
Epoch 4 loss 0.6418066063939081 valid acc 15/16
Epoch 4 loss 0.6242592444335828 valid acc 15/16
Epoch 4 loss 0.7340174756664454 valid acc 15/16
Epoch 4 loss 1.4679916281579515 valid acc 16/16
Epoch 4 loss 0.6162219726329357 valid acc 15/16
Epoch 4 loss 0.46955597548624267 valid acc 15/16
Epoch 4 loss 0.6706063186808546 valid acc 16/16
Epoch 4 loss 2.1490363194551536 valid acc 15/16
Epoch 4 loss 0.6514156519269415 valid acc 15/16
Epoch 4 loss 0.3315580721327018 valid acc 15/16
Epoch 4 loss 0.9330223097490409 valid acc 15/16
Epoch 4 loss 1.2444276170725816 valid acc 16/16
Epoch 4 loss 0.5980635194226505 valid acc 16/16
Epoch 4 loss 0.21812767597550572 valid acc 16/16
Epoch 4 loss 1.3732975175857205 valid acc 15/16
Epoch 4 loss 0.7990571506312641 valid acc 16/16
Epoch 4 loss 0.542943505769488 valid acc 15/16
Epoch 4 loss 0.8700418537375374 valid acc 16/16
Epoch 4 loss 0.5375173186786577 valid acc 15/16
Epoch 4 loss 0.6832588335244162 valid acc 15/16
Epoch 4 loss 1.3134932135290187 valid acc 15/16
Epoch 4 loss 0.31384203413545064 valid acc 15/16
Epoch 4 loss 0.9647330113096242 valid acc 16/16
Epoch 4 loss 0.8946269307941843 valid acc 16/16
Epoch 4 loss 0.6891095114160362 valid acc 16/16
Epoch 4 loss 1.4750007498126423 valid acc 16/16
Epoch 4 loss 0.8946379520916609 valid acc 16/16
Epoch 4 loss 1.2544970834846718 valid acc 16/16
Epoch 4 loss 0.5931406159697081 valid acc 16/16
Epoch 4 loss 1.1176604429136636 valid acc 15/16
Epoch 4 loss 0.7457031828156173 valid acc 16/16
Epoch 4 loss 1.1624836819967825 valid acc 16/16
Epoch 4 loss 0.5322641822108605 valid acc 16/16
Epoch 4 loss 0.724093443953466 valid acc 15/16
Epoch 4 loss 0.5827515111431243 valid acc 16/16
Epoch 4 loss 0.9072853245962508 valid acc 16/16
Epoch 5 loss 0.008726707781001763 valid acc 16/16
Epoch 5 loss 1.1318536630133258 valid acc 16/16
Epoch 5 loss 1.2249503108735007 valid acc 16/16
Epoch 5 loss 0.6964901512511483 valid acc 14/16
Epoch 5 loss 0.4613497740674043 valid acc 15/16
Epoch 5 loss 0.19540464682163902 valid acc 15/16
Epoch 5 loss 1.4593172087290107 valid acc 16/16
Epoch 5 loss 1.3837798719741436 valid acc 16/16
Epoch 5 loss 0.8305034292138049 valid acc 16/16
Epoch 5 loss 0.565969181201641 valid acc 14/16
Epoch 5 loss 0.5292857750845478 valid acc 15/16
Epoch 5 loss 1.739996075250275 valid acc 15/16
Epoch 5 loss 1.1598047343864253 valid acc 16/16
Epoch 5 loss 0.9550926877456419 valid acc 14/16
Epoch 5 loss 0.6455662682593083 valid acc 16/16
Epoch 5 loss 0.5404527322102438 valid acc 15/16
Epoch 5 loss 1.52269071303562 valid acc 15/16
Epoch 5 loss 1.2877106144449775 valid acc 15/16
Epoch 5 loss 1.0484047046412026 valid acc 15/16
Epoch 5 loss 0.47022156145200666 valid acc 15/16
Epoch 5 loss 0.57067176900594 valid acc 15/16
Epoch 5 loss 0.4934957216564764 valid acc 16/16
Epoch 5 loss 0.5382127130424852 valid acc 15/16
Epoch 5 loss 0.6550892386717186 valid acc 15/16
Epoch 5 loss 0.48921139390733154 valid acc 15/16
Epoch 5 loss 0.7650136049620148 valid acc 15/16
Epoch 5 loss 0.754179166612936 valid acc 15/16
Epoch 5 loss 0.6054633569013654 valid acc 15/16
Epoch 5 loss 0.7673036534238231 valid acc 14/16
Epoch 5 loss 0.770451740169912 valid acc 16/16
Epoch 5 loss 1.0143158105056795 valid acc 16/16
Epoch 5 loss 0.9960476434461024 valid acc 16/16
Epoch 5 loss 0.5489187401097366 valid acc 15/16
Epoch 5 loss 1.1330008395696234 valid acc 15/16
Epoch 5 loss 1.1216718141824789 valid acc 15/16
Epoch 5 loss 0.4633492607349138 valid acc 15/16
Epoch 5 loss 0.41520873244863843 valid acc 15/16
Epoch 5 loss 0.6199189527577176 valid acc 15/16
Epoch 5 loss 0.7005562883507328 valid acc 16/16
Epoch 5 loss 0.9866156250175288 valid acc 16/16
Epoch 5 loss 0.3411358426569915 valid acc 16/16
Epoch 5 loss 0.8986368452803947 valid acc 16/16
Epoch 5 loss 0.923018902012741 valid acc 15/16
Epoch 5 loss 0.6721004872829286 valid acc 15/16
Epoch 5 loss 0.5235734095794548 valid acc 16/16
Epoch 5 loss 0.17584253094740343 valid acc 16/16
Epoch 5 loss 0.6340926449540294 valid acc 16/16
Epoch 5 loss 1.083379519282123 valid acc 15/16
Epoch 5 loss 0.6770704215321723 valid acc 16/16
Epoch 5 loss 0.4846839099611696 valid acc 15/16
Epoch 5 loss 0.6919797483438362 valid acc 15/16
Epoch 5 loss 0.6618289238923508 valid acc 16/16
Epoch 5 loss 0.8090558579829097 valid acc 16/16
Epoch 5 loss 0.524623633199618 valid acc 15/16
Epoch 5 loss 0.8045458103241732 valid acc 15/16
Epoch 5 loss 0.5285753165507446 valid acc 15/16
Epoch 5 loss 1.2033484912543548 valid acc 15/16
Epoch 5 loss 1.056973662787767 valid acc 16/16
Epoch 5 loss 0.7529276643579665 valid acc 16/16
Epoch 5 loss 0.6411647751032941 valid acc 16/16
Epoch 5 loss 1.0629653674554227 valid acc 16/16
Epoch 5 loss 0.21981582054887086 valid acc 16/16
Epoch 5 loss 0.7658850271240758 valid acc 16/16
Epoch 6 loss 0.00251046051699505 valid acc 16/16
Epoch 6 loss 0.987051511557929 valid acc 16/16
Epoch 6 loss 1.3406141964850826 valid acc 16/16
Epoch 6 loss 0.7954553892872094 valid acc 16/16
Epoch 6 loss 0.23041175845654713 valid acc 15/16
Epoch 6 loss 0.6391693230942088 valid acc 15/16
Epoch 6 loss 0.8924115306974958 valid acc 15/16
Epoch 6 loss 1.1461619195544857 valid acc 16/16
Epoch 6 loss 0.1884149391763285 valid acc 16/16
Epoch 6 loss 0.4756703313845562 valid acc 16/16
Epoch 6 loss 0.3820359908913111 valid acc 16/16
Epoch 6 loss 1.7314229658511722 valid acc 16/16
Epoch 6 loss 1.3075469946791922 valid acc 16/16
Epoch 6 loss 0.5972426291447512 valid acc 16/16
Epoch 6 loss 0.7198750582892318 valid acc 16/16
Epoch 6 loss 0.676950188508394 valid acc 16/16
Epoch 6 loss 0.6453765525424553 valid acc 16/16
Epoch 6 loss 0.9433618529980548 valid acc 16/16
Epoch 6 loss 0.7569711166439095 valid acc 16/16
Epoch 6 loss 0.5346372183459418 valid acc 16/16
Epoch 6 loss 0.8463943189529242 valid acc 16/16
Epoch 6 loss 0.9278391251703757 valid acc 16/16
Epoch 6 loss 0.2183542572710301 valid acc 16/16
Epoch 6 loss 0.47092566692399407 valid acc 16/16
Epoch 6 loss 0.6180738810288418 valid acc 16/16
Epoch 6 loss 0.4225861702736569 valid acc 16/16
Epoch 6 loss 0.20324498723070442 valid acc 16/16
Epoch 6 loss 0.5612582327748387 valid acc 16/16
Epoch 6 loss 0.5679311443773389 valid acc 16/16
Epoch 6 loss 0.4882259185502561 valid acc 16/16
Epoch 6 loss 0.21388279506172891 valid acc 16/16
Epoch 6 loss 1.0169081528977406 valid acc 14/16
Epoch 6 loss 0.8102689635729048 valid acc 16/16
Epoch 6 loss 0.7979566987890042 valid acc 15/16
Epoch 6 loss 2.197856716941104 valid acc 16/16
Epoch 6 loss 0.42170265783665 valid acc 16/16
Epoch 6 loss 0.698592219941528 valid acc 16/16
Epoch 6 loss 0.5194756157524039 valid acc 16/16
Epoch 6 loss 0.5735573409500034 valid acc 16/16
Epoch 6 loss 0.45854524167803806 valid acc 16/16
Epoch 6 loss 0.3688027403883899 valid acc 16/16
Epoch 6 loss 0.5579738389120813 valid acc 16/16
Epoch 6 loss 0.6767057871148044 valid acc 16/16
Epoch 6 loss 2.224382837453968 valid acc 15/16
Epoch 6 loss 1.1490692439929147 valid acc 16/16
Epoch 6 loss 0.5533426711683254 valid acc 15/16
Epoch 6 loss 0.5633715730852938 valid acc 15/16
Epoch 6 loss 0.6961518868549816 valid acc 15/16
Epoch 6 loss 0.3406035463266536 valid acc 16/16
Epoch 6 loss 0.6504339135257164 valid acc 16/16
Epoch 6 loss 0.38063320626867125 valid acc 16/16
Epoch 6 loss 0.6362912749513787 valid acc 16/16
Epoch 6 loss 0.47789555976940823 valid acc 16/16
Epoch 6 loss 0.37765441022179497 valid acc 16/16
Epoch 6 loss 0.3787914882227516 valid acc 15/16
Epoch 6 loss 0.5149268952837399 valid acc 16/16
Epoch 6 loss 0.8555867020126265 valid acc 16/16
Epoch 6 loss 0.331650247247643 valid acc 16/16
Epoch 6 loss 1.0023149598407044 valid acc 16/16
Epoch 6 loss 0.541322447649017 valid acc 16/16
Epoch 6 loss 0.06205063340015282 valid acc 16/16
Epoch 6 loss 0.33349000944728135 valid acc 16/16
Epoch 6 loss 0.6721356174582835 valid acc 16/16
Epoch 7 loss 0.01757799577876401 valid acc 16/16
Epoch 7 loss 0.7341514956383735 valid acc 16/16
Epoch 7 loss 0.9476471778965534 valid acc 16/16
Epoch 7 loss 0.3362154175456904 valid acc 16/16
Epoch 7 loss 0.5076652300359143 valid acc 16/16
Epoch 7 loss 0.5087069708635358 valid acc 16/16
Epoch 7 loss 0.8988245715498575 valid acc 16/16
Epoch 7 loss 0.866911098943517 valid acc 16/16
Epoch 7 loss 0.7310577512587021 valid acc 16/16
Epoch 7 loss 0.36861355029121085 valid acc 16/16
Epoch 7 loss 0.30517373095715994 valid acc 16/16
Epoch 7 loss 1.2619568099087903 valid acc 16/16
Epoch 7 loss 0.9190115372088931 valid acc 16/16
Epoch 7 loss 0.7168621052698338 valid acc 16/16
Epoch 7 loss 0.5738378599638365 valid acc 16/16
Epoch 7 loss 0.8410390822391047 valid acc 16/16
Epoch 7 loss 1.0865855788878798 valid acc 16/16
Epoch 7 loss 0.8977850056351282 valid acc 16/16
Epoch 7 loss 0.6943820700685668 valid acc 16/16
Epoch 7 loss 0.25208662580320096 valid acc 16/16
Epoch 7 loss 1.186633446074295 valid acc 16/16
Epoch 7 loss 0.8353060887136745 valid acc 16/16
Epoch 7 loss 0.188046209090419 valid acc 16/16
Epoch 7 loss 0.8052150825298758 valid acc 16/16
Epoch 7 loss 0.3417681665353962 valid acc 16/16
Epoch 7 loss 0.7024700500864682 valid acc 16/16
Epoch 7 loss 0.6340426469021387 valid acc 16/16
Epoch 7 loss 0.24062479157401495 valid acc 16/16
Epoch 7 loss 0.484292286227359 valid acc 16/16
Epoch 7 loss 0.5664012659644395 valid acc 16/16
Epoch 7 loss 0.8039369443152492 valid acc 16/16
Epoch 7 loss 0.42919843362468846 valid acc 16/16
Epoch 7 loss 0.3834111962818047 valid acc 16/16
Epoch 7 loss 0.4590465289289888 valid acc 16/16
Epoch 7 loss 1.070161045682855 valid acc 16/16
Epoch 7 loss 0.45779013017814957 valid acc 16/16
Epoch 7 loss 0.32808711380301053 valid acc 16/16
Epoch 7 loss 0.4738261788125864 valid acc 16/16
Epoch 7 loss 0.49293233719205654 valid acc 16/16
Epoch 7 loss 0.47773684976332464 valid acc 16/16
Epoch 7 loss 0.34702868024532374 valid acc 16/16
Epoch 7 loss 0.772341465544956 valid acc 16/16
Epoch 7 loss 0.38725561910395434 valid acc 16/16
Epoch 7 loss 0.519599040098335 valid acc 16/16
Epoch 7 loss 0.8087679338751483 valid acc 16/16
Epoch 7 loss 0.2870531310206198 valid acc 16/16
Epoch 7 loss 0.5585513668521248 valid acc 16/16
Epoch 7 loss 1.3073409478098708 valid acc 15/16
Epoch 7 loss 0.36849481676997337 valid acc 16/16
Epoch 7 loss 0.2353085944207153 valid acc 16/16
Epoch 7 loss 0.5943256852395611 valid acc 16/16
Epoch 7 loss 0.467654498886125 valid acc 16/16
Epoch 7 loss 1.2910604910061938 valid acc 16/16
Epoch 7 loss 0.4279625642119753 valid acc 16/16
Epoch 7 loss 0.6971676347357769 valid acc 16/16
Epoch 7 loss 0.3940144682679243 valid acc 16/16
Epoch 7 loss 0.4086814560008363 valid acc 16/16
Epoch 7 loss 0.48818816712913743 valid acc 16/16
Epoch 7 loss 0.3336703880192018 valid acc 15/16
Epoch 7 loss 0.47002309239229745 valid acc 16/16
Epoch 7 loss 0.1949190946698246 valid acc 16/16
Epoch 7 loss 0.21329612677022225 valid acc 16/16
Epoch 7 loss 0.7391432537994106 valid acc 16/16
Epoch 8 loss 0.0012258615083137747 valid acc 16/16
Epoch 8 loss 0.49329178949759256 valid acc 16/16
Epoch 8 loss 0.4195232795967603 valid acc 16/16
Epoch 8 loss 0.24231565862179563 valid acc 16/16
Epoch 8 loss 0.3036825691005725 valid acc 15/16
Epoch 8 loss 0.3306672351034622 valid acc 16/16
Epoch 8 loss 0.9842945327048795 valid acc 15/16
Epoch 8 loss 1.1844273346588183 valid acc 16/16
Epoch 8 loss 0.3538545234816056 valid acc 16/16
Epoch 8 loss 0.39251926768942014 valid acc 16/16
Epoch 8 loss 0.5628800925768739 valid acc 16/16
Epoch 8 loss 0.9558245826060868 valid acc 16/16
Epoch 8 loss 0.6965505357088775 valid acc 16/16
Epoch 8 loss 0.8327153626062391 valid acc 15/16
Epoch 8 loss 0.5127583842823418 valid acc 15/16
Epoch 8 loss 0.6179379998289221 valid acc 15/16
Epoch 8 loss 0.5320025901969756 valid acc 16/16
Epoch 8 loss 1.0315849162013895 valid acc 16/16
Epoch 8 loss 0.389388889936871 valid acc 16/16
Epoch 8 loss 0.17581484522503465 valid acc 16/16
Epoch 8 loss 0.5287018085552095 valid acc 16/16
Epoch 8 loss 0.1674957269559587 valid acc 16/16
Epoch 8 loss 0.04837326242281581 valid acc 16/16
Epoch 8 loss 0.7060694621463837 valid acc 16/16
Epoch 8 loss 1.0126266542125688 valid acc 15/16
Epoch 8 loss 0.7253886489825763 valid acc 16/16
Epoch 8 loss 0.2164814215604122 valid acc 16/16
Epoch 8 loss 0.36405551618819 valid acc 15/16
Epoch 8 loss 0.24861687509508879 valid acc 16/16
Epoch 8 loss 0.05047872716507923 valid acc 16/16
Epoch 8 loss 0.49551941567152047 valid acc 16/16
Epoch 8 loss 0.2648487249489313 valid acc 16/16
Epoch 8 loss 0.2247656922418585 valid acc 16/16
Epoch 8 loss 0.2990702004699431 valid acc 15/16
Epoch 8 loss 0.5878379896216028 valid acc 16/16
Epoch 8 loss 0.8140045968332636 valid acc 16/16
Epoch 8 loss 0.400853557734551 valid acc 16/16
Epoch 8 loss 0.9505536334291929 valid acc 16/16
Epoch 8 loss 0.24295740259465287 valid acc 16/16
Epoch 8 loss 0.16258774888720828 valid acc 16/16
Epoch 8 loss 0.48567476106099344 valid acc 16/16
Epoch 8 loss 0.29632736424672884 valid acc 16/16
Epoch 8 loss 0.38862469030669317 valid acc 16/16
Epoch 8 loss 0.15721817470810948 valid acc 16/16
Epoch 8 loss 0.3828581290250874 valid acc 16/16
Epoch 8 loss 0.11591385902467677 valid acc 16/16
Epoch 8 loss 0.2115040645646541 valid acc 16/16
Epoch 8 loss 0.8061545550613303 valid acc 16/16
Epoch 8 loss 0.1291900565901138 valid acc 16/16
Epoch 8 loss 0.4128488600338228 valid acc 16/16
Epoch 8 loss 0.3586890725182514 valid acc 16/16
Epoch 8 loss 0.4919115745479263 valid acc 15/16
Epoch 8 loss 0.5629177123190036 valid acc 16/16
Epoch 8 loss 0.12455728945097005 valid acc 16/16
Epoch 8 loss 0.6504278537259882 valid acc 16/16
Epoch 8 loss 0.1781313037810915 valid acc 16/16
Epoch 8 loss 0.7913745215337267 valid acc 16/16
Epoch 8 loss 0.23021882567811724 valid acc 16/16
Epoch 8 loss 1.1357741166472293 valid acc 16/16
Epoch 8 loss 0.8362272522666523 valid acc 16/16
Epoch 8 loss 0.6338188176840867 valid acc 14/16
Epoch 8 loss 0.3157507099131637 valid acc 16/16
Epoch 8 loss 0.5096266591616346 valid acc 16/16
Epoch 9 loss 0.0015563068535688895 valid acc 16/16
Epoch 9 loss 0.46359473828402425 valid acc 16/16
Epoch 9 loss 0.7094406559268887 valid acc 16/16
Epoch 9 loss 0.5686990983016936 valid acc 15/16
Epoch 9 loss 0.1796840824037674 valid acc 16/16
Epoch 9 loss 0.20020711438134187 valid acc 15/16
Epoch 9 loss 0.7409532435901823 valid acc 15/16
Epoch 9 loss 0.24391411960031034 valid acc 16/16
Epoch 9 loss 0.43124905619194076 valid acc 16/16
Epoch 9 loss 0.27170899696579615 valid acc 16/16
Epoch 9 loss 0.4724318146528525 valid acc 16/16
Epoch 9 loss 0.7227536179596634 valid acc 16/16
Epoch 9 loss 0.5691376215046499 valid acc 16/16
Epoch 9 loss 1.063237928523245 valid acc 15/16
Epoch 9 loss 0.6827225606760576 valid acc 16/16
Epoch 9 loss 0.40682825202003897 valid acc 16/16
Epoch 9 loss 0.8407118335943702 valid acc 16/16
Epoch 9 loss 0.7911057672399373 valid acc 16/16
Epoch 9 loss 0.5480428610058927 valid acc 16/16
Epoch 9 loss 0.6139308532183358 valid acc 15/16
Epoch 9 loss 1.0188596804796655 valid acc 15/16
Epoch 9 loss 0.22751151597488195 valid acc 16/16
Epoch 9 loss 0.20099590599844785 valid acc 16/16
Epoch 9 loss 0.47758627012181015 valid acc 15/16
Epoch 9 loss 0.27621800728276485 valid acc 15/16
Epoch 9 loss 0.5612365240087255 valid acc 16/16
Epoch 9 loss 0.40423285885667165 valid acc 15/16
Epoch 9 loss 0.14160976223624971 valid acc 15/16
Epoch 9 loss 0.6495483879388355 valid acc 15/16
Epoch 9 loss 0.36581236853356003 valid acc 15/16
Epoch 9 loss 0.5579684077577035 valid acc 16/16
Epoch 9 loss 1.004170418856061 valid acc 16/16
Epoch 9 loss 0.22993057582689314 valid acc 16/16
Epoch 9 loss 0.12108980095144595 valid acc 16/16
Epoch 9 loss 0.6356247477100385 valid acc 16/16
Epoch 9 loss 0.6548804413646485 valid acc 16/16
Epoch 9 loss 0.6511908214877848 valid acc 16/16
Epoch 9 loss 0.6176825228587721 valid acc 16/16
Epoch 9 loss 0.6058887390818468 valid acc 16/16
Epoch 9 loss 0.138030031952439 valid acc 16/16
Epoch 9 loss 0.11130516141701116 valid acc 16/16
Epoch 9 loss 0.49562922992951847 valid acc 16/16
Epoch 9 loss 0.2058993472451853 valid acc 16/16
Epoch 9 loss 0.5164589139184281 valid acc 16/16
Epoch 9 loss 0.6578892023953119 valid acc 15/16
Epoch 9 loss 0.2711774195922707 valid acc 16/16
Epoch 9 loss 0.5618345096123369 valid acc 16/16
Epoch 9 loss 0.7237397444612939 valid acc 16/16
Epoch 9 loss 0.27158731486418763 valid acc 15/16
Epoch 9 loss 0.5538072009039419 valid acc 16/16
Epoch 9 loss 0.38393854595040083 valid acc 16/16
Epoch 9 loss 0.3237892273781493 valid acc 16/16
Epoch 9 loss 0.1408789932717987 valid acc 16/16
Epoch 9 loss 0.08877010093914484 valid acc 16/16
Epoch 9 loss 0.5032425117610828 valid acc 16/16
Epoch 9 loss 0.4807113579418753 valid acc 16/16
Epoch 9 loss 0.8788071111102824 valid acc 16/16
Epoch 9 loss 0.7975823565413244 valid acc 16/16
Epoch 9 loss 0.4757550425764151 valid acc 16/16
Epoch 9 loss 0.5421462521179958 valid acc 16/16
Epoch 9 loss 0.01729893175270547 valid acc 16/16
Epoch 9 loss 0.2951956309598067 valid acc 16/16
Epoch 9 loss 0.36570314759783723 valid acc 16/16
Epoch 10 loss 0.0031628362303136592 valid acc 16/16
Epoch 10 loss 0.42425930053588146 valid acc 16/16
Epoch 10 loss 0.7163867935510346 valid acc 15/16
Epoch 10 loss 0.18098888582179976 valid acc 15/16
Epoch 10 loss 0.6382475048852126 valid acc 16/16
Epoch 10 loss 0.5520943057779366 valid acc 16/16
Epoch 10 loss 0.5428076553881127 valid acc 16/16
Epoch 10 loss 0.9818998779864858 valid acc 16/16
Epoch 10 loss 0.5823832534572368 valid acc 16/16
Epoch 10 loss 0.4537703541434162 valid acc 16/16
Epoch 10 loss 0.24494162395056798 valid acc 16/16
Epoch 10 loss 0.3327612300011258 valid acc 16/16
Epoch 10 loss 0.6548320379777821 valid acc 16/16
Epoch 10 loss 0.9312745273732278 valid acc 16/16
Epoch 10 loss 0.20217756300196155 valid acc 16/16
Epoch 10 loss 0.13301465074337404 valid acc 16/16
Epoch 10 loss 0.8320099123307432 valid acc 16/16
Epoch 10 loss 0.21861079165819575 valid acc 16/16
Epoch 10 loss 0.4247501588939335 valid acc 16/16
Epoch 10 loss 0.24001533874219208 valid acc 16/16
Epoch 10 loss 0.5235407203127533 valid acc 16/16
Epoch 10 loss 0.4661047187420029 valid acc 16/16
Epoch 10 loss 0.5424579819121593 valid acc 16/16
Epoch 10 loss 0.48580405777346897 valid acc 16/16
Epoch 10 loss 0.5788516005962032 valid acc 16/16
Epoch 10 loss 0.3940082646111488 valid acc 16/16
Epoch 10 loss 0.32647055983706486 valid acc 16/16
Epoch 10 loss 0.04356556607977441 valid acc 16/16
Epoch 10 loss 0.3340676363494319 valid acc 16/16
Epoch 10 loss 0.3417830589401282 valid acc 16/16
Epoch 10 loss 0.45466746056548346 valid acc 16/16
Epoch 10 loss 0.08926885860746286 valid acc 16/16
Epoch 10 loss 0.3760437823341532 valid acc 16/16
Epoch 10 loss 0.3255905487442503 valid acc 16/16
Epoch 10 loss 0.7458877342965773 valid acc 16/16
Epoch 10 loss 0.6553005949721444 valid acc 16/16
Epoch 10 loss 0.3235087019510543 valid acc 16/16
Epoch 10 loss 0.25859635062503705 valid acc 16/16
Epoch 10 loss 0.3801172828508351 valid acc 16/16
Epoch 10 loss 0.34629571290768657 valid acc 16/16
Epoch 10 loss 0.48723261570960447 valid acc 16/16
Epoch 10 loss 0.2486299512206614 valid acc 16/16
Epoch 10 loss 0.20961001130252246 valid acc 16/16
Epoch 10 loss 0.13312889575672293 valid acc 16/16
Epoch 10 loss 0.47703262178302624 valid acc 16/16
Epoch 10 loss 0.09044771438935123 valid acc 16/16
Epoch 10 loss 0.3161648505052206 valid acc 16/16
Epoch 10 loss 0.7327247932162819 valid acc 16/16
Epoch 10 loss 0.533091675459773 valid acc 16/16
Epoch 10 loss 0.6753239860148837 valid acc 16/16
Epoch 10 loss 0.19485523726252874 valid acc 16/16
Epoch 10 loss 0.06693574129710753 valid acc 16/16
Epoch 10 loss 0.22362930220263916 valid acc 16/16
Epoch 10 loss 0.2721679150519049 valid acc 16/16
Epoch 10 loss 0.3746278260350181 valid acc 16/16
Epoch 10 loss 0.07050347027002274 valid acc 16/16
Epoch 10 loss 0.5968051615218427 valid acc 16/16
Epoch 10 loss 0.9560316980585689 valid acc 16/16
Epoch 10 loss 0.7859883335100442 valid acc 15/16
Epoch 10 loss 0.14085991075603138 valid acc 16/16
Epoch 10 loss 0.27580789725219085 valid acc 16/16
Epoch 10 loss 0.05371410352507844 valid acc 16/16
Epoch 10 loss 0.507058291032026 valid acc 16/16
Epoch 11 loss 0.0003959971033224985 valid acc 16/16
Epoch 11 loss 0.7275516419982933 valid acc 16/16
Epoch 11 loss 1.000834818664653 valid acc 15/16
Epoch 11 loss 0.2858169990905216 valid acc 15/16
Epoch 11 loss 0.4768312502281661 valid acc 16/16
Epoch 11 loss 0.1597095029571847 valid acc 16/16
Epoch 11 loss 0.736986165761198 valid acc 15/16
Epoch 11 loss 0.7998347568550653 valid acc 16/16
Epoch 11 loss 0.4905629596321785 valid acc 16/16
Epoch 11 loss 0.3860931598803093 valid acc 16/16
Epoch 11 loss 0.5589347069270858 valid acc 16/16
Epoch 11 loss 0.4903978308392202 valid acc 16/16
Epoch 11 loss 0.7300541966577246 valid acc 16/16
Epoch 11 loss 0.19760839483736187 valid acc 16/16
Epoch 11 loss 0.15487481021925806 valid acc 16/16
Epoch 11 loss 0.1383417467804889 valid acc 16/16
Epoch 11 loss 0.5536998701650997 valid acc 16/16
Epoch 11 loss 0.5539713091522309 valid acc 16/16
Epoch 11 loss 0.253175943698945 valid acc 16/16
Epoch 11 loss 0.24638409934441907 valid acc 16/16
Epoch 11 loss 0.8020993027055753 valid acc 16/16
Epoch 11 loss 0.36958181935697226 valid acc 16/16
Epoch 11 loss 0.05111234720599933 valid acc 16/16
Epoch 11 loss 0.13562254847704203 valid acc 16/16
Epoch 11 loss 0.17638639447192123 valid acc 16/16
Epoch 11 loss 0.15045083492171363 valid acc 16/16
Epoch 11 loss 0.22704623643520988 valid acc 16/16
Epoch 11 loss 0.08986695868981343 valid acc 16/16
Epoch 11 loss 0.13322075505942804 valid acc 16/16
Epoch 11 loss 0.27744066689961866 valid acc 16/16
Epoch 11 loss 0.12162976595196245 valid acc 16/16
Epoch 11 loss 0.25829407745434396 valid acc 16/16
Epoch 11 loss 0.14545590658661772 valid acc 16/16
Epoch 11 loss 0.8534496986334907 valid acc 16/16
Epoch 11 loss 0.6912105017327903 valid acc 16/16
Epoch 11 loss 0.0726920109438629 valid acc 16/16
Epoch 11 loss 0.47774578012338376 valid acc 16/16
Epoch 11 loss 0.7500512433575167 valid acc 16/16
Epoch 11 loss 0.5593197499576512 valid acc 16/16
Epoch 11 loss 0.5662097189609505 valid acc 16/16
Epoch 11 loss 0.12356584155439584 valid acc 16/16
Epoch 11 loss 0.2506948915494758 valid acc 16/16
Epoch 11 loss 0.12515219313694476 valid acc 16/16
Epoch 11 loss 0.2505292687810705 valid acc 16/16
Epoch 11 loss 0.6154061593805168 valid acc 15/16
Epoch 11 loss 0.9036866416701044 valid acc 16/16
Epoch 11 loss 0.5722931812947523 valid acc 15/16
Epoch 11 loss 0.9774064296194009 valid acc 16/16
Epoch 11 loss 0.15977048773608774 valid acc 15/16
Epoch 11 loss 0.241638092603347 valid acc 16/16
Epoch 11 loss 0.940910114182273 valid acc 16/16
Epoch 11 loss 0.19547143310046522 valid acc 16/16
Epoch 11 loss 0.20815621279684715 valid acc 16/16
Epoch 11 loss 0.0614614542820105 valid acc 16/16
Epoch 11 loss 0.7041693262332743 valid acc 16/16
Epoch 11 loss 0.18565108095074623 valid acc 16/16
Epoch 11 loss 0.5636840743198841 valid acc 16/16
Epoch 11 loss 0.15253471037360972 valid acc 16/16
Epoch 11 loss 0.29281485961222015 valid acc 16/16
Epoch 11 loss 0.5130328252572689 valid acc 16/16
Epoch 11 loss 0.3431844858507557 valid acc 15/16
Epoch 11 loss 0.05427683785732279 valid acc 15/16
Epoch 11 loss 0.5532149960934629 valid acc 16/16
Epoch 12 loss 0.0013462786893237988 valid acc 16/16
Epoch 12 loss 0.5430930774019231 valid acc 16/16
Epoch 12 loss 0.45026297117077657 valid acc 16/16
Epoch 12 loss 0.3483145278008422 valid acc 16/16
Epoch 12 loss 0.09469713072534547 valid acc 16/16
Epoch 12 loss 0.3402040143758994 valid acc 15/16
Epoch 12 loss 0.5354881017224987 valid acc 16/16
Epoch 12 loss 0.369808954924154 valid acc 16/16
Epoch 12 loss 0.1802189096059026 valid acc 16/16
Epoch 12 loss 0.276860614655761 valid acc 16/16
Epoch 12 loss 0.252494678963296 valid acc 16/16
Epoch 12 loss 0.2815920111870773 valid acc 16/16
Epoch 12 loss 0.5093438576322676 valid acc 16/16
Epoch 12 loss 0.10898794927322741 valid acc 16/16
Epoch 12 loss 0.41531870397579973 valid acc 16/16
Epoch 12 loss 0.07782313994839873 valid acc 16/16
Epoch 12 loss 0.9316656334391096 valid acc 16/16
Epoch 12 loss 0.2709699325111055 valid acc 16/16
Epoch 12 loss 0.5749424213902741 valid acc 16/16
Epoch 12 loss 0.1212174487983213 valid acc 16/16
Epoch 12 loss 0.6125899863546548 valid acc 16/16
Epoch 12 loss 0.1692635821511127 valid acc 16/16
Epoch 12 loss 0.047796387188559 valid acc 16/16
Epoch 12 loss 0.27063814181433615 valid acc 16/16
Epoch 12 loss 0.1096907138608168 valid acc 16/16
Epoch 12 loss 0.30783872782153315 valid acc 16/16
Epoch 12 loss 0.42403446656937077 valid acc 16/16
Epoch 12 loss 0.13074985677980067 valid acc 16/16
Epoch 12 loss 0.23762057658855618 valid acc 16/16
Epoch 12 loss 0.03697921745673827 valid acc 16/16
Epoch 12 loss 0.5548953466038065 valid acc 16/16
Epoch 12 loss 0.14783216912325042 valid acc 16/16
Epoch 12 loss 0.1949821998076699 valid acc 16/16
Epoch 12 loss 0.25292376476590905 valid acc 16/16
Epoch 12 loss 0.6286625317862502 valid acc 15/16
Epoch 12 loss 0.14375380406673705 valid acc 16/16
Epoch 12 loss 0.21367024893629527 valid acc 16/16
Epoch 12 loss 0.4128874503028749 valid acc 16/16
Epoch 12 loss 0.291717172541595 valid acc 16/16
Epoch 12 loss 0.36579352938480364 valid acc 16/16
Epoch 12 loss 0.25059044042819123 valid acc 16/16
Epoch 12 loss 0.2393083959225189 valid acc 16/16
Epoch 12 loss 0.6436869556886908 valid acc 16/16
Epoch 12 loss 0.12038078979037409 valid acc 16/16
Epoch 12 loss 0.6679064978195903 valid acc 16/16
Epoch 12 loss 0.15960309198408962 valid acc 16/16
Epoch 12 loss 0.07522303016984772 valid acc 16/16
Epoch 12 loss 0.8419381634606932 valid acc 16/16
Epoch 12 loss 0.11079568593557387 valid acc 16/16
Epoch 12 loss 0.14383007059531655 valid acc 16/16
Epoch 12 loss 0.13510532755438243 valid acc 16/16
Epoch 12 loss 0.12782048838297427 valid acc 16/16
Epoch 12 loss 0.05925751005580904 valid acc 16/16
Epoch 12 loss 0.4867477387075847 valid acc 16/16
Epoch 12 loss 0.4708896270047072 valid acc 16/16
Epoch 12 loss 0.6664385085122095 valid acc 16/16
Epoch 12 loss 0.41845097881702314 valid acc 16/16
Epoch 12 loss 0.2947861450985443 valid acc 16/16
Epoch 12 loss 0.24916272337372064 valid acc 16/16
Epoch 12 loss 0.15636401139096764 valid acc 16/16
Epoch 12 loss 0.08882141861895015 valid acc 16/16
Epoch 12 loss 0.2664222449027101 valid acc 16/16
Epoch 12 loss 0.16702719995931836 valid acc 16/16
Epoch 13 loss 0.039478023467585555 valid acc 16/16
Epoch 13 loss 0.2605086356001157 valid acc 16/16
Epoch 13 loss 0.4435715354955936 valid acc 16/16
Epoch 13 loss 0.1293743364775376 valid acc 16/16
Epoch 13 loss 0.13051939394599055 valid acc 16/16
Epoch 13 loss 0.6815134362797436 valid acc 16/16
Epoch 13 loss 0.3305970931388641 valid acc 16/16
Epoch 13 loss 0.2311172068601524 valid acc 16/16
Epoch 13 loss 0.3295705907024604 valid acc 16/16
Epoch 13 loss 0.16893403247302458 valid acc 16/16
Epoch 13 loss 0.14753931809882498 valid acc 16/16
Epoch 13 loss 0.7062199455163088 valid acc 16/16
Epoch 13 loss 0.6009318427874625 valid acc 16/16
Epoch 13 loss 0.7191973203818409 valid acc 16/16
Epoch 13 loss 0.39850276701097576 valid acc 16/16
Epoch 13 loss 0.23185079765239436 valid acc 16/16
Epoch 13 loss 0.472538661530798 valid acc 16/16
Epoch 13 loss 0.6538617424996567 valid acc 16/16
Epoch 13 loss 0.34210643715255173 valid acc 16/16
Epoch 13 loss 0.2748631226541381 valid acc 16/16
Epoch 13 loss 0.7926319723122638 valid acc 16/16
Epoch 13 loss 0.5475651927989046 valid acc 16/16
Epoch 13 loss 0.09917127559945443 valid acc 16/16
Epoch 13 loss 0.015000637724635335 valid acc 16/16
Epoch 13 loss 0.5595908385072047 valid acc 16/16
Epoch 13 loss 0.16865835846492866 valid acc 16/16
Epoch 13 loss 0.6110474053009831 valid acc 16/16
Epoch 13 loss 0.2715769969717422 valid acc 16/16
Epoch 13 loss 0.1554176699661235 valid acc 16/16
Epoch 13 loss 0.2342195920563071 valid acc 16/16
Epoch 13 loss 0.7120242074101235 valid acc 16/16
Epoch 13 loss 0.4960312422286721 valid acc 16/16
Epoch 13 loss 0.3179249304317036 valid acc 16/16
Epoch 13 loss 0.3667776407565708 valid acc 16/16
Epoch 13 loss 0.931142990775902 valid acc 15/16
Epoch 13 loss 0.366653606311618 valid acc 16/16
Epoch 13 loss 0.6115462062253436 valid acc 15/16
Epoch 13 loss 0.2154060355700403 valid acc 16/16
Epoch 13 loss 0.2736559283508173 valid acc 16/16
Epoch 13 loss 0.21162664716247004 valid acc 16/16
Epoch 13 loss 0.6162146280853916 valid acc 16/16
Epoch 13 loss 0.10654426285449958 valid acc 16/16
Epoch 13 loss 0.10949746169755797 valid acc 16/16
Epoch 13 loss 0.08196925561358648 valid acc 16/16
Epoch 13 loss 0.8429890883293504 valid acc 16/16
Epoch 13 loss 0.2221336520500749 valid acc 16/16
Epoch 13 loss 0.6676533441087076 valid acc 16/16
Epoch 13 loss 0.5050049997837678 valid acc 16/16
Epoch 13 loss 0.2970894036397093 valid acc 16/16
Epoch 13 loss 0.15476286740518302 valid acc 15/16
Epoch 13 loss 0.489666368409229 valid acc 16/16
Epoch 13 loss 0.26629159218053766 valid acc 16/16
Epoch 13 loss 0.3020528261489338 valid acc 15/16
Epoch 13 loss 0.12654836483720838 valid acc 15/16
Epoch 13 loss 0.5034262758604604 valid acc 15/16
Epoch 13 loss 0.10167509897740834 valid acc 15/16
Epoch 13 loss 0.10718816652122048 valid acc 16/16
Epoch 13 loss 0.1567648386949767 valid acc 16/16
Epoch 13 loss 0.43815498336719666 valid acc 16/16
Epoch 13 loss 0.11129656269711152 valid acc 16/16
Epoch 13 loss 0.032405625393591764 valid acc 16/16
Epoch 13 loss 0.09837269356820144 valid acc 16/16
Epoch 13 loss 0.4904353394297206 valid acc 16/16
Epoch 14 loss 0.014982324908427436 valid acc 16/16
Epoch 14 loss 0.1741867841530101 valid acc 16/16
Epoch 14 loss 0.2949370373144232 valid acc 16/16
Epoch 14 loss 0.13566295305740012 valid acc 16/16
Epoch 14 loss 0.11800513463643664 valid acc 15/16
Epoch 14 loss 0.9237513046896935 valid acc 16/16
Epoch 14 loss 0.6326416796560299 valid acc 16/16
Epoch 14 loss 0.2216021979058345 valid acc 16/16
Epoch 14 loss 0.2675406554069294 valid acc 16/16
Epoch 14 loss 0.10969072635948302 valid acc 16/16
Epoch 14 loss 0.4170652689547305 valid acc 16/16
Epoch 14 loss 0.2800104995866876 valid acc 16/16
Epoch 14 loss 0.3507125252020188 valid acc 16/16
Epoch 14 loss 0.19547908526444735 valid acc 16/16
Epoch 14 loss 0.38956108412026524 valid acc 16/16
Epoch 14 loss 0.197936967038851 valid acc 16/16
Epoch 14 loss 0.27361575169257124 valid acc 16/16
Epoch 14 loss 0.16312025183055529 valid acc 16/16
Epoch 14 loss 0.06423821837684729 valid acc 16/16
Epoch 14 loss 0.17071686511783307 valid acc 16/16
Epoch 14 loss 0.5926388593289017 valid acc 15/16
Epoch 14 loss 0.23938579773266694 valid acc 16/16
Epoch 14 loss 0.13780690758838793 valid acc 16/16
Epoch 14 loss 0.18089652875672635 valid acc 16/16
Epoch 14 loss 0.7174921636626896 valid acc 16/16
Epoch 14 loss 0.3660615841667472 valid acc 16/16
Epoch 14 loss 0.23906509864187073 valid acc 16/16
Epoch 14 loss 0.44749057412388404 valid acc 15/16
Epoch 14 loss 0.1899456352781138 valid acc 16/16
Epoch 14 loss 0.3838380925356023 valid acc 15/16
Epoch 14 loss 0.36772799472197737 valid acc 16/16
Epoch 14 loss 0.2031708954685883 valid acc 16/16
Epoch 14 loss 0.21264002355912234 valid acc 16/16
Epoch 14 loss 0.23645549214884895 valid acc 16/16
Epoch 14 loss 0.687120602855942 valid acc 14/16
Epoch 14 loss 0.26303617063236145 valid acc 15/16
Epoch 14 loss 0.809871148587799 valid acc 16/16
Epoch 14 loss 0.3601316646486221 valid acc 16/16
Epoch 14 loss 0.14718998222922497 valid acc 16/16
Epoch 14 loss 0.2519432359447043 valid acc 16/16
Epoch 14 loss 0.30243649002446044 valid acc 16/16
Epoch 14 loss 0.27906122339267075 valid acc 16/16
Epoch 14 loss 0.7787786518866379 valid acc 16/16
Epoch 14 loss 0.2155946202690202 valid acc 16/16
Epoch 14 loss 0.4161404123849895 valid acc 16/16
Epoch 14 loss 0.5482737794714204 valid acc 16/16
Epoch 14 loss 0.1625687948636722 valid acc 16/16
Epoch 14 loss 0.6176912993578535 valid acc 16/16
Epoch 14 loss 0.3937185580649568 valid acc 16/16
Epoch 14 loss 0.20198938950322026 valid acc 16/16
Epoch 14 loss 0.11271215666021245 valid acc 16/16
Epoch 14 loss 0.41292244189645244 valid acc 16/16
Epoch 14 loss 0.35867390633192464 valid acc 16/16
Epoch 14 loss 0.8485180304790811 valid acc 16/16
Epoch 14 loss 0.280578635406826 valid acc 16/16
Epoch 14 loss 0.12488439757118597 valid acc 16/16
Epoch 14 loss 0.42567512761519893 valid acc 16/16
Epoch 14 loss 0.1511522129543068 valid acc 16/16
Epoch 14 loss 0.5219886036910979 valid acc 16/16
Epoch 14 loss 0.8141125568475926 valid acc 16/16
Epoch 14 loss 0.07876092294103043 valid acc 16/16
Epoch 14 loss 0.08780496099406337 valid acc 16/16
Epoch 14 loss 0.17063342739694387 valid acc 16/16
Epoch 15 loss 0.00013970553784037332 valid acc 16/16
Epoch 15 loss 0.20126471260121023 valid acc 16/16
Epoch 15 loss 0.41116545745128674 valid acc 16/16
Epoch 15 loss 0.2496332996453241 valid acc 15/16
Epoch 15 loss 0.06031519767992388 valid acc 16/16
Epoch 15 loss 0.11040450907776866 valid acc 16/16
Epoch 15 loss 0.3224441130780228 valid acc 16/16
Epoch 15 loss 0.6081315785312418 valid acc 16/16
Epoch 15 loss 0.705205988743265 valid acc 15/16
Epoch 15 loss 0.23326122508973152 valid acc 15/16
Epoch 15 loss 0.15295444766740424 valid acc 16/16
Epoch 15 loss 0.26629701626933217 valid acc 16/16
Epoch 15 loss 0.4560152589006299 valid acc 15/16
Epoch 15 loss 0.1920243710043253 valid acc 16/16
Epoch 15 loss 0.14423259472990152 valid acc 16/16
Epoch 15 loss 0.3751619092467105 valid acc 16/16
Epoch 15 loss 0.2513058795392829 valid acc 16/16
Epoch 15 loss 0.7644269193303515 valid acc 16/16
Epoch 15 loss 0.278788573779539 valid acc 16/16
Epoch 15 loss 0.10516047843533677 valid acc 16/16
Epoch 15 loss 0.4135952273790906 valid acc 16/16
Epoch 15 loss 0.48013089509611895 valid acc 16/16
Epoch 15 loss 0.2285668291469824 valid acc 16/16
Epoch 15 loss 0.2025125322252184 valid acc 16/16
Epoch 15 loss 0.3692634035668996 valid acc 16/16
Epoch 15 loss 0.05154948527874531 valid acc 16/16
Epoch 15 loss 0.7800466095013201 valid acc 15/16
Epoch 15 loss 0.4077352696638277 valid acc 15/16
Epoch 15 loss 0.3419240555723553 valid acc 15/16
Epoch 15 loss 0.1185768923654405 valid acc 16/16
Epoch 15 loss 0.09229300718556877 valid acc 15/16
Epoch 15 loss 0.13248330323570812 valid acc 15/16
Epoch 15 loss 0.4331770666480738 valid acc 16/16
Epoch 15 loss 0.23849022699388792 valid acc 15/16
Epoch 15 loss 0.5872425571989934 valid acc 16/16
Epoch 15 loss 0.2997469953448273 valid acc 15/16
Epoch 15 loss 0.3732348354185041 valid acc 16/16
Epoch 15 loss 0.5032651755884407 valid acc 16/16
Epoch 15 loss 0.31515913910815724 valid acc 16/16
Epoch 15 loss 0.0572233300358056 valid acc 16/16
Epoch 15 loss 0.529527261175651 valid acc 16/16
Epoch 15 loss 0.048030413985307385 valid acc 16/16
Epoch 15 loss 0.4831696215804354 valid acc 16/16
Epoch 15 loss 0.12207864689462883 valid acc 16/16
Epoch 15 loss 0.3227691007335754 valid acc 16/16
Epoch 15 loss 0.03230056401421072 valid acc 16/16
Epoch 15 loss 0.7016399973620489 valid acc 16/16
Epoch 15 loss 0.627477943305708 valid acc 15/16
Epoch 15 loss 0.3849865705846499 valid acc 16/16
Epoch 15 loss 0.23827262032087193 valid acc 16/16
Epoch 15 loss 0.546817378640787 valid acc 16/16
Epoch 15 loss 0.5060933577244282 valid acc 16/16
Epoch 15 loss 0.28863397040255534 valid acc 16/16
Epoch 15 loss 0.08893796642401719 valid acc 16/16
Epoch 15 loss 0.5303443466470373 valid acc 16/16
Epoch 15 loss 0.14294986965104706 valid acc 16/16
Epoch 15 loss 0.07074374118267734 valid acc 16/16
Epoch 15 loss 0.15666683074063648 valid acc 16/16
Epoch 15 loss 0.6807388491453256 valid acc 16/16
Epoch 15 loss 1.1574256383073727 valid acc 16/16
Epoch 15 loss 0.34210547881142767 valid acc 16/16
Epoch 15 loss 0.4475893678455288 valid acc 16/16
Epoch 15 loss 0.40089566567984 valid acc 16/16
Epoch 16 loss 0.1671279208275197 valid acc 16/16
Epoch 16 loss 0.3353348563961305 valid acc 16/16
Epoch 16 loss 0.2019922725170043 valid acc 16/16
Epoch 16 loss 0.21003187401929563 valid acc 16/16
Epoch 16 loss 0.7518792662405023 valid acc 15/16
Epoch 16 loss 0.1303898794285452 valid acc 16/16
Epoch 16 loss 0.5367923665699321 valid acc 16/16
Epoch 16 loss 0.24191696991605072 valid acc 16/16
Epoch 16 loss 0.5901738368492477 valid acc 16/16
Epoch 16 loss 0.25592458854194483 valid acc 16/16
Epoch 16 loss 0.3206424503525206 valid acc 16/16
Epoch 16 loss 0.5260258192441453 valid acc 16/16
Epoch 16 loss 0.6473151586519543 valid acc 16/16
Epoch 16 loss 0.4535515459836276 valid acc 16/16
Epoch 16 loss 0.3384521865159424 valid acc 16/16
Epoch 16 loss 0.13858847544044373 valid acc 16/16
Epoch 16 loss 0.5818326031665153 valid acc 16/16
Epoch 16 loss 0.42165774461885774 valid acc 16/16
Epoch 16 loss 0.618450739967129 valid acc 16/16
Epoch 16 loss 0.11613904326820304 valid acc 16/16
Epoch 16 loss 0.6658236549233023 valid acc 16/16
Epoch 16 loss 0.34618332264865104 valid acc 16/16
Epoch 16 loss 0.11419920948265114 valid acc 16/16
Epoch 16 loss 0.28070226689548916 valid acc 16/16
Epoch 16 loss 0.21795782098386596 valid acc 16/16
Epoch 16 loss 0.30923058530019953 valid acc 16/16
Epoch 16 loss 0.5860293052808839 valid acc 16/16
Epoch 16 loss 0.20290371616269032 valid acc 16/16
Epoch 16 loss 0.15711207844364833 valid acc 16/16
Epoch 16 loss 0.08252520686258602 valid acc 16/16
Epoch 16 loss 0.24063227906166973 valid acc 16/16
Epoch 16 loss 0.08832324563500005 valid acc 16/16
Epoch 16 loss 0.0510873977826134 valid acc 16/16
Epoch 16 loss 0.4025590709643886 valid acc 16/16
Epoch 16 loss 0.5875287010296524 valid acc 16/16
Epoch 16 loss 0.07422138164464154 valid acc 16/16
Epoch 16 loss 0.1742935183254327 valid acc 16/16
Epoch 16 loss 0.3534302500121926 valid acc 16/16
Epoch 16 loss 0.2274534673565519 valid acc 16/16
Epoch 16 loss 0.2907658296133358 valid acc 16/16
Epoch 16 loss 0.40808777511014604 valid acc 16/16
Epoch 16 loss 0.03057490527057627 valid acc 16/16
Epoch 16 loss 0.3455448805928696 valid acc 16/16
Epoch 16 loss 0.04928243590209869 valid acc 16/16
Epoch 16 loss 0.3004794180900679 valid acc 16/16
Epoch 16 loss 0.07825476910819307 valid acc 16/16
Epoch 16 loss 0.2790780928001557 valid acc 16/16
Epoch 16 loss 0.27023827797154637 valid acc 16/16
Epoch 16 loss 0.2467393587361047 valid acc 16/16
Epoch 16 loss 0.02552432089615536 valid acc 16/16
Epoch 16 loss 0.1264562896728113 valid acc 16/16
Epoch 16 loss 0.17429054272196992 valid acc 16/16
Epoch 16 loss 0.0931451566969953 valid acc 16/16
Epoch 16 loss 0.1550226104725937 valid acc 16/16
Epoch 16 loss 0.05271594820926839 valid acc 16/16
Epoch 16 loss 0.0885921003486379 valid acc 16/16
Epoch 16 loss 0.5226153311494652 valid acc 16/16
Epoch 16 loss 0.1384718121825858 valid acc 16/16
Epoch 16 loss 0.1747578494574413 valid acc 16/16
Epoch 16 loss 0.057784654869467156 valid acc 16/16
Epoch 16 loss 0.05501716945697421 valid acc 16/16
Epoch 16 loss 0.17208645713847864 valid acc 16/16
Epoch 16 loss 0.2438606312667841 valid acc 16/16
```
