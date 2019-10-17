# Prediction of Future Jamuna River Bank Planform Images from Landsat Data using Encoder-Decoder Conv-LSTM 

**Objective :** Jamuna River goes through mild to drastic changes resulting in bank erosion and deposition in various parts of it’s reach to gain stability. The most common methods of predicting riverbank line shifting requires a lot of data including water level, discharge, bathymetry data, satellite data, bankline data, sediment data. In most prediction methods the availability of the current data is not ensured, the predicted results do not capture the total non-linearity of the problem and the further predictions are made into the future the more inaccurate they become. In this thesis, an attempt has been made to formulate the riverbank line prediction as a spatiotemporal sequence forecasting problem in which both the input and the predicted target are spatiotemporal sequences. 

<br/>

**Methods :** For the Jamuna River planform prediction task Seq2Seq LSTM based Convolutional Autoencoder neural network architecture was explored. The effects of adding skip connections from Conv encoder activation maps to Conv decoder activation maps were also looked into. The resulting prediction model could take four historical 256×256 resolution greyscale NBRT Landsat images taken over the Jamuna River, with an interval of one year between each image, as inputs and predict the river planform images for the next two years with an interval of one year between the two.

<br/>

**Results :**

![alt text](https://github.com/antorhasan/BScthesis_bank_image_prediction/blob/master/pngs/org1.png)
<p align="center">
  <b>Actual Image One Year into the Future</b><br>
</p>

<br>

![alt text](https://github.com/antorhasan/BScthesis_bank_image_prediction/blob/master/pngs/pre1.png)
<p align="center">
  <b>Predicted Image One Year into the Future</b><br>
</p>

<br>
<br>
<br>

![alt text](https://github.com/antorhasan/BScthesis_bank_image_prediction/blob/master/pngs/org2.png)
<p align="center">
  <b>Actual Image Two Years into the Future</b><br>
</p>

<br>

![alt text](https://github.com/antorhasan/BScthesis_bank_image_prediction/blob/master/pngs/pre2.png)
<p align="center">
  <b>Predicted Image Two Years into the Future</b><br>
</p>

<br>
