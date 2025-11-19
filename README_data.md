# Dataset Card for Representational Stability Fictional Data

## Dataset Description
* Repository: [GitHub Repository](https://github.com/samanthadies/representational_stability)
* Paper: [Representational Stability of Truth in Large Language Models]()
* Point of Contact: [Samantha Dies](mailto:dies.s@northeastern.edu)

### Dataset Summary

The **Representational Stability** fictional dataset is made to supplement the 
**Trilemma of Truth** dataset ([here](https://huggingface.co/datasets/carlomarxx/trilemma-of-truth)).

The **Trilemma of Truth** data contains three types of statements:
* Factually **true** statements
* Factually **false** statements
* **Synthetic**, neither-valued statements generated to mimic statements ***unseen*** during LLM training

The **Representational Stability** fictional dataset adds new types of statements:
* **Fictional**, neither-valued statements generated to mimic statements ***seen*** during LLM training, but in a fictional, non-real-world context

The three files correspond to three different domains of statements:
* ```cities_loc_fictional.csv```: statements about city-country relations
* ```med_indications_fictional.csv```: drug-indication associations
* ```defs_fictional.csv```: synonym, type, and instance relationships from lexical knowledge

Each dataset contains a mix of **affirmative** and **negated** statements utilizing fictional entities.

### Statement Types

Even though our fictional statements are neither-true-nor-false within a real-world context,
we annotate each of them as canonically-true or canonically-false depending on its truth
value within the canonical fictional context. As such, we have four configurations:
* Canonically-true and affirmative
* Canonically-true and negated
* Canonically-false and affirmative
* Canonically-false and negated

### Statement Examples

* **City Locations** (`cities_loc`):
  * "The city of Bikini Bottom is located in the Pacific Ocean." (canonically-true, affirmative)
  * "The city of Arendelle is not located in Rohan." (canonically-true, negated)
  * "The city of Neo-Tokyo is located in Maine." (canonically-false, affirmative)
  * "The Emerald City is not located in Oz." (canonically-false, negated)
* **Medical Indications** (`med_indications`):
  * "The Trump Virus is indicated for the treatment of Xenovirus Takis-B." (canonically-true, affirmative)
  * "Cryostim is not indicated for the treatment of Dragon Pox." (canonically-true, negated)
  * "Novril is indicated for the treatment of Dryditch Fever." (canonically-false, affirmative)
  * "Gurdyroot is not indicated for the treatment of Gulping Plimpies." (canonically-false, negated)
* **Word Definitions** (`defs`):
  * "Snoivi is a type of hammock." (canonically-true, affirmative)
  * "Whoppsy-whiffling is not a type of food." (canonically-true, negated)
  * "Koakte is a type of plant." (canonically-false, affirmative)
  * "Utumauti is not a type of fruit." (canonically-false, negated)

### Paper

This dataset is introduced in:

    @article{dies2025representationalstability,
      title={Representational Stability in Large Language Models},
      author={Samantha Dies and Maynard Courtney and Germans Savcisens and Tina Eliassi-Rad},
      journal={},
      doi={},
      year={2025},
    }

In the paper, we combine this data with the 
[Trilemma of Truth dataset](https://huggingface.co/datasets/carlomarxx/trilemma-of-truth) 
and describe the motivation, data-collection pipeline, evaluation protocol, 
and evaluation on popular open-source LLMs. See the [full text on arXiv]() for the 
methodology and results.

### Supported Tasks

* `text-classification`, `zero-shot-prompting`: The dataset can be used to train a probe 
for veracity tracking (e.g., identifying true statements, false statements, and 
neither-valued statements) when there are different types of neither statements 
(i.e., fictional are familiar, synthetic are unfamiliar to the LLMs).
* `question-answering`: The dataset can be used to evaluate an LLM for factual knowledge,
particularly when neither-value statements are included.

### Fields

Each dataset consists of a `statement` that includes `object_1` and `object_2`.
Depending on the combination of objects, the statement could be `correct` (i.e.,
canonically-true; if the statement is not correct, `correct_object_2` specifies
the object that would make the statement correct). Statements could also be negated
(`negation==True`).

Data splits used in [the paper]() are denoted with the `in_train`, `in_test`, and
`in_cal` columns. The `in_cal` column can be used for either calibration or validation,
depending on the experimental setup.

```md
{'statement': 'The city of Bikini Bottom is located in Maine.',
 'object_1': 'Bikini Bottom',
 'object_2': 'Maine',
 'correct_object_2': 'Pacific Ocean',
 'correct': False,
 'negation': False,
 'real_object': False,
 'fake_object': False,
 'fictional_object': True,
 'category': cities,
 'in_train': 1,
 'in_test': 0,
 'in_cal': 0
}
```

### Data Splits

| Dataset                   | Train  |  Calibration |  Test |  Total |
|---------------------------|--------|--------------|-------|--------|
| cities_loc_fictional      | 4746   | 1772         | 2229  | 8747   |
| med_indications_fictional | 4636   | 1721         | 2121  | 8478   |
| defs_fictional            | 6488   | 2514         | 3041  | 12043  |

The split ratio is about 55% train / 20% calibration / 25% test

### Dataset Sources

City Locations:
* [List of Fictional Settlements (Wikipedia)](https://en.wikipedia.org/wiki/List_of_fictional_settlements)
* [List of Fictional City-States (Wikipedia)](https://en.wikipedia.org/wiki/List_of_fictional_city-states_in_literature)

Medical Indications:
* [Fandom NeoEncyclopedia - List of Fictional Diseases](https://neoencyclopedia.fandom.com/wiki/List_of_fictional_diseases)
* [Fandom NeoEncyclopedia - List of Fictional Toxins](https://neoencyclopedia.fandom.com/wiki/List_of_fictional_toxins)
* [ChemEurope's List of Fictional Medicine and Drugs](https://www.chemeurope.com/en/encyclopedia/List_of_fictional_medicines_and_drugs.html)
* [The Thackery T. Lambshed Pocket Guide to Eccentric & Discredited Diseases](https://archive.org/details/thackerytlambshe0000unse)

Word Definitions:
* [Gobblefunk (Roald Dahl)](https://beelinguapp.com/blog/Dahl%20Dictionary:%20A%20List%20of%20103%20Words%20Made-up%20By%20Roald%20Dahl)
* [Dothraki (Schleitwiler, P. & Shuflin, G)](https://conlang.org/language-creation-conference/lcc5/1-dothraki-initial-text/)
* [Na'vi (Avatar Wiki)](https://dict-navi.com/en/dictionary/list/?type=classification&ID=1)


### Citations

If you use this dataset, please cite the original authors as listed in the [GitHub Repository](https://github.com/samanthadies/representational_stability).


arXiv Preprint:

    @article{dies2025representationalstability,
      title={Representational Stability in Large Language Models},
      author={Samantha Dies and Maynard Courtney and Germans Savcisens and Tina Eliassi-Rad},
      journal={},
      doi={},
      year={2025},
    }

Trilemma of Truth Dataset:

    @misc{trilemma2025data,
      title={trilemma-of-truth (Revision cd49e0e)},
      author={Germans Savcisens and Tina Eliassi-Rad},
      url={https://huggingface.co/datasets/carlomarxx/trilemma-of-truth},
      doi={10.57967/hf/5900},
      publisher={HuggingFace}
      year={2025},
    }

Trilemma of Truth Paper:

    @inproceedings{savcisens2025trilemma,
      title={Trilemma of Truth in Large Language Models},
      author={Savcisens, Germans and Eliassi-Rad, Tina},
      booktitle={Mechanistic Interpretability Workshop at Neur{IPS} 2025},
      year={2025},
      note={\url{https://openreview.net/forum?id=z7dLG2ycRf}},
    }