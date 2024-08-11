# Sign-Detect

Code for detecting and tracking Hand Signers (Indian Sign Language) in News Video. Please see [todo](www.abc.com] for more details

## Installation

Please ensure you have `conda` installed and be prepared with the full path of your installation for which you'll be prompted. In my case, when I tested it on an A10 G on Lambda Labs, the path to my `conda` installation was `/home/ubuntu/miniconda3`. 

Start by installing `mmpose`. The instructions for doing so can be found [here](https://mmpose.readthedocs.io/en/latest/installation.html). In short, you have to prepare a `conda` environment with name `openmmlab`. Then you have to install all the dependencies of `mmpose` and then the repository itself. Once this is done, use the following to run our detector on test videos.

```
conda activate openmmlab
python3 -m pip install -r requirements.txt
bash run_test.sh
```

## License

## References 

```
@article{Ratner_2017,
   title={Snorkel: rapid training data creation with weak supervision},
   volume={11},
   ISSN={2150-8097},
   url={http://dx.doi.org/10.14778/3157794.3157797},
   DOI={10.14778/3157794.3157797},
   number={3},
   journal={Proceedings of the VLDB Endowment},
   publisher={Association for Computing Machinery (ACM)},
   author={Ratner, Alexander and Bach, Stephen H. and Ehrenberg, Henry and Fries, Jason and Wu, Sen and Ré, Christopher},
   year={2017},
   month=nov, pages={269–282} }
```

```
@misc{mmpose2020,
    title={OpenMMLab Pose Estimation Toolbox and Benchmark},
    author={MMPose Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmpose}},
    year={2020}
}
```

```
@InProceedings{Albanie2020bsl1k,
    author       = "Samuel Albanie and G{\"u}l Varol and Liliane Momeni and Triantafyllos Afouras and Joon Son Chung and Neil Fox and Andrew Zisserman",
    title        = "{BSL-1K}: {S}caling up co-articulated sign language recognition using mouthing cues",
    booktitle    = "European Conference on Computer Vision",
    year         = "2020",
}
```

## Acknowledgement

We used [nanoGPT](https://github.com/karpathy/nanoGPT) for building our classifier.
