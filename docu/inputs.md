## How to train on your own data

### Easy way

The easiest way is to provide data in a similar way to the voc data. To do that create files `train` and `val` similar to [train.lst](../data/train.lst). Each line of this file is supposed to contain a path to an image and a path to the corresponding ground truth. 

The ground truth file is assumed to be an image. You can configure those colours in the `hype` file by changing

```
  "classes": [
        {
            "name": "background",
            "colors": [
                "/#000000"
            ]
        },
        {
            "name": "bicycle",
            "colors": [
                "/#008000"
            ]
        },
        {
            "name": "aeroplane",
            "colors": [
                "/#800000"
            ]
        },
        {
            "name": "bird",
            "colors": [
                "/#808000"
            ]
        },
        {
            "name": "boat",
            "colors": [
                "/#000080"
            ]
        },
        {
            "name": "bottle",
            "colors": [
                "/#800080"
            ]
        },
        {
            "name": "bus",
            "colors": [
                "/#008080"
            ]
        },
        {
            "name": "chair",
            "colors": [
                "/#C00000"
            ]
        },
        {
            "name": "car",
            "colors": [
                "/#808080"
            ]
        },
        {
            "name": "cat",
            "colors": [
                "/#400000"
            ]
        },
        {
            "name": "cow",
            "colors": [
                "/#408000"
            ]
        },
        {
            "name": "diningtable",
            "colors": [
                "/#C08000"
            ]
        },
        {
            "name": "dog",
            "colors": [
                "/#400080"
            ]
        },
        {
            "name": "horse",
            "colors": [
                "/#C00080"
            ]
        },
        {
            "name": "motorbike",
            "colors": [
                "/#408080"
            ]
        },
        {
            "name": "person",
            "colors": [
                "/#C08080"
            ]
        },
        {
            "name": "pottedplant",
            "colors": [
                "/#004000"
            ]
        },
        {
            "name": "sheep",
            "colors": [
                "/#804000"
            ]
        },
        {
            "name": "sofa",
            "colors": [
                "/#00C000"
            ]
        },
        {
            "name": "train",
            "colors": [
                "/#80C000"
            ]
        },
        {
            "name": "tvmonitor",
            "colors": [
                "/#004080"
            ]
        }