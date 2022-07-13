<br />
<div align="center">
  <h3 align="center">His Master's Voice: <br>
    Audio localisation using deep neural networks</h3>
  </p><p align="center">
  
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
As part of the Data Science Retreat Batch 30 final portfolio project, we set to determine whether deep neural networks could be trained to determine the location of a sound. <br>
<br>
The inspiration for this supervised machine learning problem, came from the 2022 paper by Andrew Francl and Josh McDermott from M.I.T., in which they have aimed to simulate human hearing using deep neural networks. This allowed them to prove that human hearing is specially adapted to our environment, to help us localise sound.

What motivated us to pursue this idea was two-fold:<br>
1) Audio has received relatively little attention in comparison to images,<br>
2) The set-up by Francl and McDermott made use of Cochleagramms, a specifically developed package based on Matplot. We wanted to determine whether this process can be simplified.


The project involved generating artificial data using Ableton 11's Dear Reality plug-in.
Additionally, live recordings were made using the MNIST Spoken Digits and a randomly shuffled
composition of the Librispeech Tensorflow datasets. 
The sound files were then cut into 1-2 second .wav files and a spectrogram was made. 
These spectrograms were then used to train the neural network.

After the neural network was trained, we attempted to test the model in a live environment. For this purpose we have constructed a dummy with two microphones. A sound would be made from a given direction and the model would have to predict the location on a 3D graphic of the room.

<p align="right">(<a href="#top">back to top</a>)</p>

### Built With
The project was implemented using the following packages:

* TensorFlow
* Tensorflow_io
* MatplotLib (PyPlot and Animation)
* OS
* Pydub
* sounddevice
* queue
* sys
* Numpy
* Pickle
* Glob
<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

### Data Collection
We have generated data from two sources, an artificial source and a live-source. 

#### Artificial Data

For the artificial data we have made use of Sennheiser's Dear Reality Plug-In for Ableton. With this plug in, we were able to set Elevation and Azimuth (direction).
For our first attempt we have generated 24 different categories of data based on azimuth using a N, NE, E, SE, S convention. In other words, we have split the "room" into eight sectors according to compass direction and the azimuth increased by 45 degrees for each sector.
A recording of elevation of 45 degrees upwards, 0 degrees, and 45 degrees downwards was made for each sector.

The sound file used for the recording was the spoken digits MNIST Tensorflow dataset, which can be obtained from the following link:
https://www.tensorflow.org/datasets/catalog/spoken_digit

#### Live Data

For the live recording samples, we have set up a dummy with two microphones, to simulate ears, in the middle of a room. 
The room was accordingly divided into 8 sectors using the NWSE split. We have then used a bluetooth speaker along with a telescopic
tripod to direct the sound towards the dummy.

For the initial live-recordings, we have used the spoken digits MNIST Tensorflow dataset. To train the neural networks
on more complex audio, we have also composed a shuffled track using the librisspeech Tensorflow set, which can be obtained
from the following site:

https://www.tensorflow.org/datasets/catalog/librispeech

Each data source, artificial MNIST, live-MNIST and live-Librisspeech was then cut into 1-2 second sample .wav files, producing 3000 samples for each elevation-direction combination.

<div align=""center">
<img src = data/images/data_balance.png alt="Logo" width="700" height="500">
</div>

### Prerequisites

Ensure that the required packages listed under "Built with" have been installed and are up to date.


<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

The Francl and McDermott paper was tested using ten different neural network architectures,
that provided them the best results for their hearing emulation. The architecture was
coded for optimal use with Tensorflow 1. For the purposes of this implementation, we have
adapted their codes for the purposes of Tensorflow 2.

The models have been trained using an Adam Optimizer with a final Softmax activation
layer in order to obtain the probabilities for the Sectors and Elevation. It should be
noted, that given Elevation/ Sectors had different accuracies with different models,
as a result, rather than having a multiple target output with one model, we ran
separate models for direction and elevation.

<!-- LICENSE -->
## License

MIT License

Copyright (c) 2022 Al-Gurnawi and Buzar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

We would like to thank the following:

1) Andrew Francl and Josh McDermott for providing us with the inspiration
for this project,
2) Jose Quesada and the team at the Data Science Retreat for training us
3) Last, but not least, Dr. Tristan Behrens for being our mentor and guiding us.

<p align="right">(<a href="#top">back to top</a>)</p>