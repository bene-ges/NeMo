# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""
This script can be used to create a more class balanced file from a set of the data files of the English Google Text Normalization dataset
Of all the input files in `input_dir` this script takes the first file and computes class counts in it.
Then it predicts a sampling frequency, which is needed to .

For those that are underrepresented, quantitatively defined as lower than `min_number`, the other files are scanned for sentences that have the missing patterns.
Those sentences are appended to the first file and outputted.

USAGE Example:
1. Download the Google TN dataset from https://www.kaggle.com/google-nlu/text-normalization
2. Unzip the English subset (e.g., by running `tar zxvf  en_with_types.tgz`). Then there will a folder named `en_with_types`.
3. Run the data_split.py, data_preprocessing.py scripts to obtain cleaned data files
4. Run this script on the training data portion
# python upsample.py       \
        --input_dir=ru_with_types_preprocessed_data_split/test_preprocessed
        --output_file=ru_with_types_preprocessed_data_split/test_preprocessed/test.tsv
        --min_number=8000

In this example, the final file will be test.tsv.
ATTENTION: to comply with old format you need to additionally replace PUNCT and VERBATIM 3-d column to "sil"
   awk 'BEGIN {FS="\t"}($1 == "PUNCT"){print $1 "\t" $2 "\tsil"}($1 == "VERBATIM"){print $1 "\t" $2 "\tsil"} ($1 != "PUNCT" && $1 != "VERBATIM"){print $0}' < test.tsv > test2.tsv
"""

import glob
from argparse import ArgumentParser
from collections import defaultdict

parser = ArgumentParser(description="Russian Text Normalization upsampling")
parser.add_argument("--input_dir", required=True, type=str, help='Path to input directory with preprocessed data')
parser.add_argument("--output_file", required=True, type=str, help='Path to output file')
parser.add_argument("--min_number", default=2000, type=int, help='minimum number per pattern')
args = parser.parse_args()

cls2count = defaultdict(int)

out = open(args.output_file, "w", encoding="utf-8")
input_files = sorted(glob.glob(f"{args.input_dir}/output-*"))
for fn in input_files:
    print("Processing: ", fn)
    with open(fn, "r", encoding="utf-8") as f:
        contents = f.read()
        contents_parts = contents.split("<eos>\t<eos>\n")
        for sent in contents_parts:
            take_this_sent = False
            sent_parts = sent.split("\n")
            for sent_part in sent_parts:
                fields = sent_part.split("\t")
                cls = fields[0]
                if cls2count[cls] < args.min_number:
                    take_this_sent = True
                    break
            if take_this_sent:
                for sent_part in sent_parts: #update counts
                    fields = sent_part.split("\t")
                    cls = fields[0]
                    cls2count[cls] += 1
                out.write(sent + "<eos>\t<eos>\n")
print(cls2count)

out.close()
