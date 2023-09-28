# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
import os

import datasets


_CITATION = """\
@misc{Johnson.2019,
 abstract = {The MIMIC Chest X-ray (MIMIC-CXR) Database v1.0.0 is a large publicly available dataset of chest radiographs with structured labels. The dataset contains 371,920 images corresponding to 224,548 radiographic studies performed at the Beth Israel Deaconess Medical Center in Boston, MA. The dataset is de-identified to satisfy the US Health Insurance Portability and Accountability Act of 1996 (HIPAA) Safe Harbor requirements. Protected health information (PHI) has been removed. The dataset is intended to support a wide body of research in medicine including image understanding, natural language processing, and decision support.},
 author = {Johnson, Alistair E. W. and Pollard, Tom and Mark, Roger and Berkowitz, Seth and Horng, Steven},
 date = {2019},
 title = {The MIMIC-CXR Database},
 keywords = {Radiography, Thoracic},
 publisher = {physionet.org},
 doi = {10.13026/C2JT1Q}
}
"""


_DESCRIPTION = """\
This script is designed to load local images on a server stored in the structure defined by the Dataset creators.
The script should also combine text with the images to allow the training of a image text stable diffusion model
After signing the data usage agreement, being a credentialed user on physionet and completing the required trainings, the user can open URL below.
"""


_HOMEPAGE ="https://mimic-cxr.mit.edu/" 


_LICENSE = "https://physionet.org/content/mimic-cxr/view-license/2.0.0/"


_URLS = {
    "first_domain":["/work/srankl/thesis/development/modelDesign_bias_CXR/diffusers/huggingface_dataset"] #r"C:\Users\rankl\Documents\uni\Thesis\Development\modelDesign_bias_CXR\diffusers\huggingface_dataset"[r"C:\Users\rankl\Documents\uni\Thesis\Development\modelDesign_bias_CXR\data\MIMICCXR\huggingface_dataset"]
}



class MIMICCXR(datasets.GeneratorBasedBuilder):
    """Builder Config for MIMICCXR"""
    VERSION = datasets.Version("1.1.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')

    DEFAULT_CONFIG_NAME = "first_domain"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        # TODO: This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
       
        features = datasets.Features(
            {
                "image": datasets.Image(),
                "text": datasets.Value("string")
                # These are the features of your dataset like images, labels ...
            }
        )
        
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        urls = _URLS[self.config.name]
        data_dir = urls[0] #dl_manager.extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "train.jsonl"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "validate.jsonl"),
                    "split": "dev",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "test.jsonl"),
                    "split": "test"
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        with open(filepath, encoding="utf-8") as f:
            for idx, row in enumerate(f):
                data = json.loads(row)
                if self.config.name == "first_domain":
                    image_path = data["image"]
                    with open(image_path, "rb") as image_file:
                        image_bytes = image_file.read()
                    yield split +"_"+ str(idx), {
                        "image": {"path": image_path, "bytes": image_bytes},
                        "text": data["text"],
                    }

