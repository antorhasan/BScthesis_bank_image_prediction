# UNDERGRADUATE THESIS :Prediction of Future Jamuna River Bank Planform Images from Landsat Data using Encoder-Decoder Conv-LSTM 

**Objective :** Jamuna River goes through mild to drastic changes resulting in bank erosion and deposition in various parts of it’s reach to gain stability. The most common methods of predicting riverbank line shifting requires a lot of data including water level, discharge, bathymetry data, satellite data, bankline data, sediment data. In most prediction methods the availability of the current data is not ensured, the predicted results do not capture the total non-linearity of the problem and the further predictions are made into the future the more inaccurate they become. In this thesis, an attempt has been made to formulate the riverbank line prediction as a spatiotemporal sequence forecasting problem in which both the input and the predicted target are spatiotemporal sequences. 

<br>

**Methods :** For the Jamuna River planform prediction task Seq2Seq LSTM based Convolutional Autoencoder neural network architecture was explored. The effects of adding skip connections from Conv encoder activation maps to Conv decoder activation maps were also looked into. The resulting prediction model could take four historical 256×256 resolution greyscale NBRT Landsat images taken over the Jamuna River, with an interval of one year between each image, as inputs and predict the river planform images for the next two years with an interval of one year between the two.

<br>

**Results :**

![alt text](https://github.com/antorhasan/BScthesis_bank_image_prediction/blob/master/pngs/org1.png)
<p align="center">
  <b>Actual Image One Year into the Future</b><br>
</p>



![alt text](https://github.com/antorhasan/BScthesis_bank_image_prediction/blob/master/pngs/pre1.png)
<p align="center">
  <b>Predicted Image One Year into the Future</b><br>
</p>

<br>
<br>


![alt text](https://github.com/antorhasan/BScthesis_bank_image_prediction/blob/master/pngs/org2.png)
<p align="center">
  <b>Actual Image Two Years into the Future</b><br>
</p>



![alt text](https://github.com/antorhasan/BScthesis_bank_image_prediction/blob/master/pngs/pre2.png)
<p align="center">
  <b>Predicted Image Two Years into the Future</b><br>
</p>

<br>

**Conclusions :** The resulting riverbank planform prediction model could easily be used for any reach of the river. It can also be used for other rivers given proper data preprocessing is carried out as most learned features are universal for all rivers. The study also suggests that spatiotemporal features of natural phenomena related to rivers can be learned by neural networks.

Even though, skip connections from convolutional encoder to convolutional decoder is thought to be helpful in increasing the resolution and quality of predicted river planform, in practical learning scenario for this domain, the skip connections proved to be inefficient. Experiments were carried out and it was observed that eventually non skip connection network learned to produce better images when trained for longer epochs.

The use of sliding window to create a dataset comprising batches of six images with a stride of one image was explored. This approach of data batching was thought to have both pros and cons. The model was assumed to be benefited from having more data instances. The drawback for this approach was thought to be that the target images of one instance would be fed as input in some other instances. From experiments presented in this study it was found that sliding window approach ends up not helping bringing the overall cost down as much as non-sliding window approach.
