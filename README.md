# Cataract_Detection_with_XAI

## Table of Content:

- [About The Web-App](#about-the-web-app)
- [Interface Design](#interface-design)
- [Technologies](#technologies)
- [Setup](#setup)
- [Research Papers](#research-papers)
- [Credits](#credits)


## About The Web-App
Cataract Detection With XAI(Explainable AI) is an Web-app for detecting if a patient has a cataract eye using fundus images with key feature of XAI that explains why a eye has a cataract or why a eye doesnot have cataract by highlighting the region of the eye.

## Interface Design
 ![image](https://github.com/harshshah37/Cataract_Detection_with_XAI/blob/main/Web_Design/266943593-81b3f49d-5370-4f54-ad90-6ce1c2b096ca.png?raw=true)
Landing page of our project.

 ![image](https://github.com/harshshah37/Cataract_Detection_with_XAI/blob/main/Web_Design/266943858-f596417d-76f5-4973-92b5-b0de78aab9bf.png?raw=true)
This part of the website describes the motivation and objective of our project.

![image](https://github.com/harshshah37/Cataract_Detection_with_XAI/blob/main/Web_Design/266944017-f39a1add-9b07-495e-b181-f583e7839677.png?raw=true)
Fundus Eye image as input (Cataract).

 ![image](https://github.com/harshshah37/Cataract_Detection_with_XAI/blob/main/Web_Design/266944055-eff0e303-9603-4066-bb35-40df09afb015.png?raw=true)
Firstly, it predicts whether it is a cataract or a normal eye. Then it gives the explanation of its output using Grad-CAM XAI by generating heatmaps.
Same is true for normal eye image

## Technologies
`Python`, `PyTorch`, `XAI Labraries`, `Pillow`, `Streamlit`

## Setup
- download or clone the repository
- download .h5 from [drive](https://drive.google.com/file/d/1i_0mLAkRwUojQA-YOuPz52c1JP-9kiVw/view?usp=drive_link)
- add this file inside /multipage/pages
- run 1_Home.py using streamlit run <pathname>

## Research Papers
- XAI meets Ophthalmology: An Explainable Approach to Cataract Detection using VGG-19 and Grad-CAM ([IEEE Link](https://ieeexplore-ieee-org.proxyiub.uits.iu.edu/document/10450053))
- Exploring the Potentials of Explainable AI for Early Cataract Detection to Foster Accessible Healthcare ([JUSST Link](https://jusst.org/exploring-the-potentials-of-explainable-ai-for-early-cataract-detection-to-foster-accessible-healthcare/))

## Credits
List of contriubutors:
- Prof. Harshal Dalvi
- Sanchit Hegde
- Rutwik Patel
