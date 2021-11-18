
# Machine learning access through API
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D SECONDARY TEST!
***
***
***
*** To avoid retyping too much info. Do a search and replace for the following:
*** github_username, repo_name, twitter_handle, email, project_title, project_description
-->

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use. TEST
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
[![twitter][twitter-shield]][twitter-url]



<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/kristopy/AI-API">
    <img src="assets/pngkey.com-particle-png-5104887.png" alt="Logo" width="250">
  </a>

  <h3 align="center">Machine learning access through API</h3>

  <p align="center">
    Number recognition and spam/ham detection implementations
    <br />
    <a href="https://github.com/kristopy/AI-API"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/kristopy/AI-API/issues">Report Bug</a>
    ·
    <a href="https://github.com/kristopy/AI-API/issues">Request Feature</a>
  </p>
</p>

<!-- ABOUT THE PROJECT -->
## About The Project

**This section contains informations about the project and how to take it to use.**

> First agenda is what the project contains. 

<!-- Working tree-->

### Working tree
```
├── Datasets
│   ├── Datasets_NUM_REC
│   │   ├── Exports
│   │   │   ├── Num_Rec_Metadata.json
│   │   │   └── Num_Rec_Model.h5
│   │   ├── NUM_REC_Classifier
│   │   │   ├── Test_datasets
│   │   │   │   ├── num-rec-test-image-dataset
│   │   │   │   └── num-rec-test-labels-dataset
│   │   │   └── Train_datasets
│   │   │       ├── num-rec-train-image-dataset
│   │   │       └──num-rec-train-labels-dataset
│   │   └── Zips
│   │       ├── num-rec-test-image-dataset
│   │       ├── num-rec-test-labels-dataset
│   │       ├── num-rec-train-image-dataset
│   │       └── num-rec-train-labels-dataset
│   └── Datasets_SMS
│       ├── Exports
│       │   ├── Spam-Metadata.json
│       │   ├── Spam-Metadata.pkl
│       │   ├── Spam-Tokenizer.json
│       │   ├── Spam_Dataset.csv
│       │   └── Spam_Model.h5
│       ├── Spam-Classifier
│       │   ├── Sms-Spam
│       │   └── Youtube-Spam
│       └── Zips
│           ├── sms-spam-dataset.zip
│           └── youtube-spam-dataset.zip
├── NUM_REC
│   ├── 1.\ Download\ Datasets\ &\ Unzip.ipynb
│   └── 2.\ Creating\ AI-model\ using\ TensorFlow.ipynb
├── README.md
├── SMS-SPAM
│   ├── 1\ -\ Download\ Datasets.ipynb
│   ├── 2\ -\ Download\ Datasets\ &\ Unzip.ipynb
│   ├── 3\ -\ Extract,\ Review\ &\ Combine\ Datasets.ipynb
│   ├── 4.\ Convert\ Datasets\ Into\ Vectors.ipynb
│   ├── 5.\ Convert\ Datasets\ Into\ Vectors,\ Split\ &\ Export.ipynb
│   └── 6.\ Creating\ Machine\ Learning\ Algorithm.ipynb
├── app
│   ├── AI_models.py
│   ├── Images_address.py
│   ├── TEST.py
│   ├── __inti.py__
│   └── main.py
├── assets
│   └── pngkey.com-particle-png-5104887.png
├── dockerfile
└── requirements.txt
```


### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/kristopy/AI-API
   ```

2. >Note: Create a virtuel envirement before installing dependencies - cd to working parent directory AI-API. 

3. For creating a virtuel envirement run:  
    ```
    $ pipenv shell
    ```

4. The first thing to do is to install the required requirements, containing all the neccesary packages in this project. 
    ```zsh
    $ pip3 install -r requirements.txt
    ```
5. >Run main.py using uvicorn.
    
    From command line run the following command:

    ```
    $ uvicorn app.main:app --reload
    ```

    The ```--reload``` parameter reloads the app when changes to the filestructure is maid. 

    Remember that inside main.py we declear the app using: 

    ```py
    app = FastAPI()
    ```

    The expected output should be: 

    ```py
    INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
    INFO:     Started reloader process [5940] using statreload
    INFO:     Started server process [5942]
    INFO:     Waiting for application startup.
    INFO:     Application startup complete.
    ```


>The fastAPI is now ready for usage. 


## Project information

> This project is two-fold - Two machine learnign algorithms based upon **Ham/Spam detection** and **Number recognition**

In the folders **`NUM_REC`** and **`SMS-SPAM`** jupyter notebook is utilized for machine learning and file-management. 

> These scripts will create folders where the Datasets are stored - as seen in working tree. 






<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/kristopy/repo.svg?style=for-the-badge
[contributors-url]: https://github.com/kristopy/AI-API/contributors
[forks-shield]: https://img.shields.io/github/forks/kristopy/repo.svg?style=for-the-badge
[forks-url]: https://github.com/kristopy/AI-API//network/members
[stars-shield]: https://img.shields.io/github/stars/kristopy/repo.svg?style=for-the-badge
[stars-url]: https://github.com/kristopy/AI-API//stargazers
[issues-shield]: https://img.shields.io/github/issues/kristopy/repo.svg?style=for-the-badge
[issues-url]: https://github.com/kristopy/AI-API//issues
<!-- [license-shield]: https://img.shields.io/github/license/kristopy/repo.svg?style=for-the-badge
[license-url]: https://github.com/kristopy/RaspberryPi_Real-Time-monitoring/blob/master/LICENSE.txt -->
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/kristofferwv
[twitter-shield]: https://img.shields.io/badge/-twitter-black.svg?style=for-the-badge&logo=twitter&colorB=555
[twitter-url]: https://twitter.com/KristofferWV
