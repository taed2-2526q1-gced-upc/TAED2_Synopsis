<!--# For reference on dataset card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/datasetcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/datasets-cards -->

# Dataset Card for CNN-DailyMail News Text Summarization

<!-- Provide a quick summary of the dataset. -->

The dataset consists news articles collected from CNN and Daily Mail, along with its summaries, intended for news summarization.

## Dataset Details

### Dataset Description

<!-- Provide a longer summary of what this dataset is. -->

This dataset contains news articles from CNN (April 2007 - April 2015) and Daily Mail (June 2010 - April 2015), along with its corresponding summary. The dataset was originally designed for question answering tasks, but later versions were made for news summarization.

- **Curated by:** Gowri Shankar Penugonda
- **Shared by:** Gowri Shankar Penugonda
- **Language(s) (NLP):** English
- **License:** CC0: Public Domain

### Dataset Sources

<!-- Provide the basic links for the dataset. -->

- **Repository:** https://cs.nyu.edu/~kcho/DMQA/

## Uses

<!-- Address questions around how the dataset is intended to be used. -->

### Direct Use

<!-- This section describes suitable use cases for the dataset. -->

This dataset can be used for Natural Language Processing (NLP) tasks, particularly for news summarization. It can also be used for analyzing content from CNN and Dailymail news, such as coverage and potential biases.

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the dataset will not work well for. -->

This dataset is not intended for tasks requiring anonymized personal data. Models trained with this dataset will not work for different languages or for summarizing texts that do not have the same style as the news. The dataset is also not intended to generate legal, medical or other professional advice.

## Dataset Structure

<!-- This section provides a description of the dataset fields, and additional information about the dataset structure such as criteria used to create the splits, relationships between data points, etc. -->

The dataset contains 3 columns, an unique id for each news, the body of the news article and the summarization or highlights of the article. The dataset has been splitted into a train, test and validation split, each one containing 287113, 13368 and 11490 rows each respectively. The mean token count for the articles are 781 and for the highlights are 56. The id for each news is a heximal formated SHA1 hashof the url where the article was retrieved.

## Dataset Creation

### Curation Rationale

<!-- Motivation for the creation of this dataset. -->

According to the curator of this dataset, the original goal was to support supervised neural methods for machine reading and question answering by providing a large amount of real natural language training data, including 313k unique articles and nearly 1M Cloze-style questions. In later versions, the curator adapted the dataset to support text summarization tasks rather than question answering.

### Source Data

<!-- This section describes the source data (e.g. news text and headlines, social media posts, translated sentences, ...). -->
The dataset consists of news articles paired with highlight sentences, which serve as summaries of the articles. The CNN articles were written between April 2007 and April 2015, and the Daily Mail articles were written between June 2010 and April 2015.

#### Data Collection and Processing

<!-- This section describes the data collection and processing process such as data selection criteria, filtering and normalization methods, tools and libraries used, etc. -->

The articles were downloaded using archives of <www.cnn.com> and <www.dailymail.co.uk> on the Wayback Machine. Tokenization and preprocessing were performed using scripts provided by the original dataset authors: Hermann et al.’s tokenization script, and See et al.’s PTBTokenizer, which lowercases the text and adds periods to lines missing them. The dataset was then, split into training, validation and test sets.

#### Who are the source data producers?

<!-- This section describes the people or systems who originally created the data. It should also include self-reported demographic or identity information for the source data creators if this information is available. -->

The articles of this dataset were written by CNN and DailyMail.

### Annotations

<!-- If the dataset contains annotations which are not part of the initial data collection, use this section to describe them. -->

#### Annotation process

<!-- This section describes the annotation process such as annotation tools used in the process, the amount of data annotated, annotation guidelines provided to the annotators, interannotator statistics, annotation validation, etc. -->

The curator created a summarization column by concatenating highlight sentences from each article to form a summary.

#### Who are the annotators?

<!-- This section describes the people or systems who created the annotations. -->

The annotations were created by the curator of this dataset.

#### Personal and Sensitive Information

<!-- State whether the dataset contains data that might be considered personal, sensitive, or private (e.g., data that reveals addresses, uniquely identifiable names or aliases, racial or ethnic origins, sexual orientations, religious beliefs, political opinions, financial or health data, etc.). If efforts were made to anonymize the data, describe the anonymization process. -->

The dataset is not anonymized as individuals names can be found in the articles. 

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

The dataset consists of news articles from CNN and the Daily Mail, which may result in differences in editorial policies, political views in articles and other biases. Because of this, language, topics and perspectives represented in the dataset may contain some bias.

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

There should be some considerations when training models. Users should evaluate the models for bias and consider augmenting the dataset with other news articles to mitigate it.

## Citation

<!-- If there is a paper or blog post introducing the dataset, the APA and Bibtex information for that should go in this section. -->

**APA:**

Gowri, S. P. CNN-DailyMail News Text Summarization. Kaggle. https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail/data.



