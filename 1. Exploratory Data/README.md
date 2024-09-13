# AI, Decision-Making and Society Problem Set #1






## Overview

The goal of this homework is to provide some hands-on experience of exploratory data analysis and data curation. We will work step-by-step through investigating a dataset, training some ML models, and evaluating the models. We will apply a minimal version of the Grounded Theory approach to define custom categories and consider the effects of different design decisions on model performance. Overall, the homework is divided into 5 parts:

1.   Spinning Up:  Prepare the environment and load the dataset.
2.   Quantitative Evaluation: Train a simple model and evaluate its performance.
3.   Exploratory Analysis: Analyze the dataset, improving it through manual cleaning and modifications.
4.   Custom Training & Evaluation: Conduct prompt engineering to explore the capabilities of large language models.
5.   Reflections on the Process: Reflect on the entire process and discuss the results.

This notebook is designed to be run in a Google Colab environment.

**Submission Instructions**: Please submit a PDF of your completed notebook to Gradescope. Make sure that all cell outputs are included. This assignment is due by 11:59 PM on Wednesday, 9/18/24.  

## Spinning Up

In this section, we will set up our environment for the rest of the assignment. We will:

*   Set up your colab to interact with the Gemini language model
*   Download a dataset to use for the assignment
*   Implement code to train a classifier

### Setting up your coding environment

First, we will set up the python environment to interact with libraries for model training and data manipulation.
Moreover, make sure to enable the GPU on the notebook by navigating to the 'runtime' tab, then clicking on 'change runtime type' and then selecting one of the available GPUs

## Install the generative AI interface
!pip install -U -q google-generativeai
!pip install transformers

## Imports in order to call relevant libraries
import re
import tqdm
import keras
import numpy as np
import pandas as pd
import os

import google.generativeai as genai

from google.colab import userdata

import seaborn as sns
import matplotlib.pyplot as plt

from keras import layers
from matplotlib.ticker import MaxNLocator
import sklearn.metrics as skmetrics

Next, we will need to configure our code to connect to the language model server. You can do this with a Colab Secret named `GOOGLE_API_KEY`. If you don't already have this configured (or you're unsure if you do) follow the instructions in the [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) quickstart.

API_KEY=userdata.get('GOOGLE_API_KEY')
genai.configure(api_key=API_KEY)

### Dataset

Now that we have configured our environment, we need some data to get started. For this assignment, we will use the [Employee Review dataset](https://www.kaggle.com/datasets/fiodarryzhykau/employee-review/code). This is a dataset of 880 employee performance reviews with a combined metric that asseses the performance & potential of the employees, measured on a scale from 1 to 9. We will consider the potential use of AI systems trained on this data.

### Getting the Data

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/fiodarryzhykau/employee-review/code) and extract the csv files train_set.csv and test_set.csv to your local machine. Then, upload it to the Colab environment.

You can find instructions for uploading a file to a colab session [here](https://saturncloud.io/blog/how-to-use-google-colab-to-work-with-local-files/#:~:text=Uploading%20Files%20to%20Google%20Colab&text=Click%20on%20the%20%E2%80%9CFiles%E2%80%9D%20tab,for%20the%20upload%20to%20complete.).

## Question 1: Looking at the Data and Documentation

Most data has documentation that describes how it was collected, what its intended purposes were, and known issues or risks. (If your data doesn't have this, its generally good practice to ask why!) Before we interact with the data, look at the documentation for this dataset and some example entries on [Kaggle](https://www.kaggle.com/datasets/fiodarryzhykau/employee-review/data). Answer the following questions with 1-3 sentences each.

#### Q 1.1: How was this dataset constructed?

**The dataset was collected from Amazon MTurk workers who were asked to create fake employee performance reviews based on a "9-box" model. The data was designed to provide variability and quality in the reviews, with clear instructions for workers to avoid using specific words directly from the given categories. A portion of the feedback was manually reviewed to ensure consistency between the review content and the assigned categories.**

#### Q 1.2: What was the original purpose for the dataset?

**The purpose was to conduct research for the Stanford NLU course to compare the performance of deep learning transfer models with more traditional models used in sentiment analysis. Its intention was to explore how well deep learning models could analyze raw, unrefined, and partially unverified data.**

#### Q 1.3: What does it mean that the dataset is "partially reviewed"? Why might this be important?

**It means that only a portion (~70%) of the records were checked for consistency, ensuring that the feedback provided by MTurk workers is consistent with the categories they were assigned. Models trained on unreviewed data might learn from incorrect or mislabeled examples, lowering accuracy or making biased model predictions. By reviewing the data, models are ensured to be accurate, which improves the reliability of the model.**

#### Q 1.4: Identify and describe one potentially appropriate and one potentially inappropriate application of the dataset (or a model trained on it). These applications should be hypothetical and do not need to directly correspond to a specific real-world use case of this dataset.

**Appropriate application: train models for multidimensional sentiment analysis, specifically in workplace performance evaluations. The dataset is designed to mimic real reviews in a controlled manner, making it suitable for developing models that could help HR teams identify employee performance and potential.**

**Inappropriate application: make automated decisions about real employees in critical HR processes such as hiring or promotion. Because the data consists of imaginary employee reviews, it does not fully represent the complexities and nuances of real employee feedback, making it risky to rely on it for high-stakes decisions. In addition, the data is only partially verified, and relying on it could propagate labeling errors into real-world outcomes.**

#### Related Reading

[Datasheets for Datasets](https://arxiv.org/abs/1803.09010) articulates the motivation and reasoning behind this documentation.

Now that we've considered the data source, let's look at the data! We'll do that by reading the .csv files into a [pandas](https://pandas.pydata.org/) dataframe (using the command pd.read_csv) and display the first few rows of the dataframe.

train_df = pd.read_csv('/train_set.csv')
train_df.set_index('id', inplace=True)

test_df = pd.read_csv('/test_set.csv')
test_df.set_index('id', inplace=True)

from google.colab import drive
drive.mount('/content/drive')

Run this cell to see an example of what a data point from the training set looks like.

test_df.head()

## Question 2: Define and train the classifier

In this section we will take the feedback review and will generate the embeddings, from which we will try to predict the performance/potential score for each employee.
The generation of the embedding is done similarly to the next text classification [example](https://colab.research.google.com/github/google-gemini/cookbook/blob/main/examples/Classify_text_with_embeddings.ipynb#scrollTo=_mwJYXpElYJc).

### Creating the embeddings

Create the embeddings using the embedding-001 model, similarly to the provided classification example. Make sure that your colab notebook is enabled with the GPU configuration.

###### Remark: the embedding generation might take a while, so be patient.

from tqdm.auto import tqdm
tqdm.pandas()

from google.api_core import retry

def make_embed_text_fn(model):

  @retry.Retry(timeout=300.0)
  def embed_fn(text: str) -> list[float]:
    # Set the task_type to CLASSIFICATION.
    embedding = genai.embed_content(model=model,
                                    content=text,
                                    task_type="classification")
    return embedding['embedding']

  return embed_fn

def create_embeddings(model, df):
  df['Embeddings'] = df['feedback'].progress_apply(make_embed_text_fn(model))
  return df

model_embeddings = 'models/embedding-001'
df_train = create_embeddings(model_embeddings, train_df)
df_test = create_embeddings(model_embeddings, test_df)

Now, if we look at the data we can see that there is a new column that includes the embeddings for the review. There are opaque numbers that show how a model represents a particular example. We will use these embeddings to train a custom classifer next.

train_df.head()

test_df.head()

## Question 3: Build a simple classification model

Following the standard classifier example, we will now build a classifier comprised of two fully-connected layers, and will use it to classify the different performance/potential metric based on the feedback embeddings.

def build_classification_model(input_size: int, num_classes: int) -> keras.Model:
  inputs = x = keras.Input(shape=(input_size,))
  x = layers.Dense(input_size, activation='relu')(x)
  x = layers.Dense(num_classes, activation='sigmoid')(x)
  return keras.Model(inputs=[inputs], outputs=x)

# Derive the embedding size from the first training element.
embedding_size = len(df_train['Embeddings'].iloc[0])

# Give your model a different name, as you have already used the variable name 'model'
classifier = build_classification_model(embedding_size, len(df_train['label'].unique()))
classifier.summary()

classifier.compile(loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   optimizer = keras.optimizers.Adam(learning_rate=0.0001),
                   metrics=['accuracy'])

### Question 3.1: Training the model

In order to train this model, we need to set the training data for our classifier. Modify the code below so that the training data are stored in the following variables:

```
y_train, x_train
```
and the validation data in
```
y_val, x_val
```
Note that, similarly to the reference [example](https://colab.research.google.com/github/google-gemini/cookbook/blob/main/examples/Classify_text_with_embeddings.ipynb#scrollTo=_mwJYXpElYJc), you might find it useful to use the function "np.stack" for the concatenation of the embeddings.


Furthermore, feel free to try several different values of NUM_EPOCHS and BATCH_SIZE to see how they affect the model's performance.

NUM_EPOCHS = 40
BATCH_SIZE = 8

# Configure the training data to fit the classifier.
y_train = df_train['label']
x_train = np.stack(df_train['Embeddings'])
y_val   = df_test['label']
x_val   = np.stack(df_test['Embeddings'])

# Train the model for the desired number of epochs.
history = classifier.fit(x=x_train,
                         y=y_train,
                         validation_data=(x_val, y_val),
                         batch_size=BATCH_SIZE,
                         epochs=NUM_EPOCHS,)

### Question 3.2: Evaluating the model

We will use Keras' <a href="https://www.tensorflow.org/api_docs/python/tf/keras/Model#evaluate"><code>Model.evaluate</code></a> procedure to get the loss and the accuracy on the test dataset.

## Evaluate classifier on validation data
classifier.evaluate(x=x_val, y=y_val, return_dict=True)

When evaluating model performance, we need to consider the possibility that our evaluation got lucky. One way to approach this is through the use of confidence intervals. These tell us a range of possible accuracy values that are consistent with the data (see a more detailed description in the the supportive PDF file). Calculate the 95% confidence interval for the accuracy of the model using the formula

\begin{equation}
    \mathrm{CI} = \mathrm{accuracy} \pm 1.96 \times \sqrt{\frac{\mathrm{accuracy} \times (1 - \mathrm{accuracy})}{\mathrm{test \ size}}}
\end{equation}

def build_CI(y_hat, y_true):
    # The function should return the one-sided width of the confidence interval, as described in the equation above.
    cor_predictions = np.sum(y_hat == y_true)
    tot_predictions = len(y_true)
    accuracy = cor_predictions / tot_predictions
    standard_error = np.sqrt(accuracy * (1 - accuracy) / tot_predictions)
    z_value = 1.96  # 95% confidence level
    ci_onesided_width = z_value * standard_error ##
    return ci_onesided_width #

y_hat_test = classifier.predict(x=x_val)
y_hat_test = np.argmax(y_hat_test, axis=1)
print('========== CIs ==========')
initial_CI_onesided_width = build_CI(y_hat_test, y_val)
print(initial_CI_onesided_width)

It is often useful to look at performance over the course of training. The next cell gives code to create this plot.

def plot_history(history):
  """
    Plotting training and validation learning curves.

    Args:
      history: model history with all the metric measures
  """
  fig, (ax1, ax2) = plt.subplots(1,2)
  fig.set_size_inches(20, 8)

  # Plot loss
  ax1.set_title('Loss')
  ax1.plot(history.history['loss'], label = 'train')
  ax1.plot(history.history['val_loss'], label = 'test')
  ax1.set_ylabel('Loss')

  ax1.set_xlabel('Epoch')
  ax1.legend(['Train', 'Validation'])

  # Plot accuracy
  ax2.set_title('Accuracy')
  ax2.plot(history.history['accuracy'],  label = 'train')
  ax2.plot(history.history['val_accuracy'], label = 'test')
  ax2.set_ylabel('Accuracy')
  ax2.set_xlabel('Epoch')
  ax2.legend(['Train', 'Validation'])

  plt.show()

plot_history(history)

#### Question 3.3

Why is there a gap between the accuracy on the training set and the accuracy on the validation set? Why does the training loss decrease while the validation loss stagnates?

**The gap between training accuracy and validation accuracy indicates overfitting. The model is learning patterns specific to the training set but is not generalizing well to the validation set, which leads to higher training accuracy and lower validation accuracy. The training loss decreases because the model is optimizing itself to fit the training data better over time. However, the validation loss stagnates because the model is not improving on unseen data, suggesting it is overfitting and memorizing the training data rather than learning general features applicable to new data.**

#### Question 3.4 - Confusion Matrix [Only for students enrolled in the graduate version of the class]

Similarly to the classical classification [example](https://colab.research.google.com/github/google-gemini/cookbook/blob/main/examples/Classify_text_with_embeddings.ipynb#scrollTo=_mwJYXpElYJc), build and observe the confusion matrix, which characterize the model accuracy across the difference classes

y_hat = classifier.predict(x=x_val)
y_hat = np.argmax(y_hat, axis=1)

labels_dict = dict(zip(df_test['nine_box_category'], df_test['label']))
labels_dict_reversed = dict(zip(df_test['label'], df_test['nine_box_category']))

labels_dict

cm = skmetrics.confusion_matrix(y_val, y_hat)
disp = skmetrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=labels_dict.keys())
disp.plot(xticks_rotation='vertical')
plt.title('Confusion matrix for newsgroup test dataset');
plt.grid(False)

What does the confusion matrix reveal about the model's performance? Are there any specific types of errors you can identify?

**The confusion matrix shows that the model performs well in certain categories, such as "Category 9: Star" and "Category 1: Risk," where most predictions align with the true labels. However, there are notable misclassifications, particularly between similar categories. For instance, "Category 4: Inconsistent Player" is often confused with "Category 2: Average Performer," indicating that the model struggles to differentiate between categories with moderate performance and potential. Additionally, there is noticeable overlap in predictions for higher-potential categories like "Category 7: Potential Gem" and "Category 8: High Potential," suggesting that the model finds it difficult to distinguish between different levels of potential when performance levels are similar. These errors imply that the model may not fully capture the nuanced differences between adjacent performance and potential categories.**

## Question 4: Data Exploration

After looking at the quantitative evaluation of the model, now we will look at qualitative analysis. This will involve exploring the data and identifying some custom categories and labels. To start, let's add a new column to the dataset with the predicted labels.

# Stack the embeddings to create x_train and x_test
x_train = np.stack(df_train['Embeddings'].values)
x_test = np.stack(df_test['Embeddings'].values)

# Predict on the training data
y_hat_train = classifier.predict(x_train)
y_hat_train = np.argmax(y_hat_train, axis=1)
df_train['predicted_label'] = y_hat_train

# Predict on the test data
y_hat_test = classifier.predict(x_test)
y_hat_test = np.argmax(y_hat_test, axis=1)
df_test['predicted_label'] = y_hat_test

#### Question 4.1: Custom Ratings

In multiple cases (and also in our dataset), some of the annotated labels might be noisy or incorrect. To that end, we would like to utilize the classifier we have developed and a human annotator to re-generate some labels. Please implement a short function that lets the user decide on the right label, and records the user choice as the new label. Then, use this function to extend the dataset and create a new set of labels. Your loop should display the confidence interval of model accuracy measured against the new labels.


You can label as many as you want, but you should label at least 30 examples.

# Function to manually label each example
pd.set_option('display.max_colwidth', None)

def manual_labeling(row):
    """
    Function to manually label data. Shows the data for review, collects the
    updated response, and then returns either the original label or the updated
    label, depending on the response.
    """

    print(f"Feedback is: {row['feedback']}")
    print(f"Original Label: {labels_dict_reversed[row['label']]}")
    print(f"Predicted Label: {labels_dict_reversed[row['predicted_label']]}")

    # Ask the user for manual input on which label is correct
    correct_label = input("Enter the correct label (or press Enter to keep the original label): ")

    if correct_label == "":  # If no input, keep the original label
        return row['label']
    else:  # If the user provides a new label, return the updated label (subtract 1 for 0 indexing)
        return int(correct_label) - 1

# Function to calculate confidence interval width based on new labels
def build_CI(y_hat, y_true):
    # Calculate accuracy
    cor_predictions = np.sum(y_hat == y_true)
    tot_predictions = len(y_true)
    accuracy = cor_predictions / tot_predictions
    # Standard error
    standard_error = np.sqrt(accuracy * (1 - accuracy) / tot_predictions)
    z_value = 1.96  # 95% confidence level
    # Return one-sided width of confidence interval
    return z_value * standard_error

# Initialize manual_label columns to -1 (indicating not labeled yet)
df_train['manual_label'] = -1
df_test['manual_label'] = -1

# Number of examples to inspect manually
examples_to_inspect = 30

# Loop through the dataset and manually update labels
for i in range(len(df_train)):

    # Apply manual labeling for df_train
    df_train.iloc[i, df_train.columns.get_loc('manual_label')] = manual_labeling(df_train.iloc[i])

    # Apply manual labeling for df_test (optional, depending on your needs)
    # df_test.iloc[i, df_test.columns.get_loc('manual_label')] = manual_labeling(df_test.iloc[i])

    # Calculate new confidence intervals using updated manual labels
    # Here we assume that we are predicting again based on the model output
    y_hat_train = df_train['predicted_label'].values
    y_true_train = df_train['manual_label'].values

    # Only calculate CI for rows that have been manually labeled
    labeled_indices = y_true_train != -1
    Curr_CI_onesided_width = build_CI(y_hat_train[labeled_indices], y_true_train[labeled_indices])

    # Print current CI every 10 examples
    if i % 10 == 0:
        print('=======================================================')
        print(f"========= Current i is {i}")
        print(f"Current CI: {Curr_CI_onesided_width}")
        print(f"Original CI: {initial_CI_onesided_width}")

    # Stop after labeling the specified number of examples
    if i >= examples_to_inspect:
        break

# Display the updated DataFrame
print("\nUpdated DataFrame with manual labels:")
print(df_train.head())



#### Question 4.2 Reflection

How well do your labels agree with the labels in the dataset? Are there any patterns to the differences? How might a model trained on the new labels be different?

Please write 3-5 sentences.

**The manually labeled examples largely agree with the predicted and original labels in this dataset. This suggests that the model is fairly accurate in predicting categories like 'Risk' (Low performance, Low potential), given the clear patterns in the feedback. However, some minor discrepancies might still exist, particularly in edge cases where feedback is ambiguous or subjective. If trained on the new labels, the model may improve its understanding of nuanced differences in feedback tone or specific performance-related language. This might lead to better handling of subtle shifts in sentiment and performance across different feedback styles.**

## Question 5: Creating New Labels

So far, over the course of the assignment, we have moved to interact more and more with the actual data:

1.   First, you looked at the data documentation and information about the dataset.
2.   Second, you looked an quantitative evaluation of a model using the existing labels.
3.   Third, you did a manual subjective evaluation of the model using your own interpretation of the labels.

Now, we'll go one step further and invent some new categories for the data. The first step is to do initial coding of the data. For this step, we will collect some free form annotations of the data.

*Hint*: If you're not sure how to go about annotating data, look at the content in Ch. 5 of Charmaz about initial coding.



### Question 5.1

First, modify the code below to collect annotations of the data and store them in the dataframe.

# Add new columns to hold the annotations. Initialize with an empty string
df_train['annotation'] = ''
df_test['annotation'] = ''

def annotate(row):
    """
    This function allows for free-form annotations of the data by asking the user
    to review each example and provide an annotation.
    """
    print(f"Feedback is: {row['feedback']}")
    print(f"Original Label: {labels_dict_reversed[row['label']]}")
    print(f"Predicted Label: {labels_dict_reversed[row['predicted_label']]}")

    # Ask the user for an annotation for the example
    annotation = input("Enter an annotation for this feedback: ")

    # Return the annotation provided by the user
    return annotation

# Apply the function to your dataset
# Apply the manual annotation function to each row and store the results
examples_to_inspect = 4  # Define how many examples you want to annotate

for i in range(len(df_train)):

    # Annotate an example from the training set
    df_train.iloc[i, df_train.columns.get_loc('annotation')] = annotate(df_train.iloc[i])

    # Annotate an example from the testing set (optional, if you want to annotate test data too)
    df_test.iloc[i, df_test.columns.get_loc('annotation')] = annotate(df_test.iloc[i])

    if i >= examples_to_inspect:
        break

    # Print the current iteration index
    if i % 3 == 0:
        print('=======================================================')
        print(f"=========Current i is {i}=========")

# Display the collected Annotations
for _, row in df_train.iterrows():
    if row['annotation'] != '':
        print(f"Feedback is: {row['feedback']}")
        print(f"Original Label: {labels_dict_reversed[row['label']]}")
        print(f"Predicted Label: {labels_dict_reversed[row['predicted_label']]}")
        print(f"Annotation: {row['annotation']}")
        print('\n')

### Question 5.2

Now that we have our annotations, the next step is to move from initial coding to focused coding. Review your annotations and identify 3 interesting similarities or differences that you found in the data. Next propose 2 different ways of categorizing the data based on the similarities.

For example, your annotations may have noticed that some reviews focus on interpersonal behavior (e.g., how well the employee works on a team) while others focus on work quality (e.g., do they finish their work on time and in a quality manner). In this case, you might create a category called "Review Focus" with labels: "other", "interpersonal skills", "work quality".

For each category please:

1.   Provide a 1-2 sentence summary.
2.   List the labels in that category.
3.   Provide an example of each label.
4.   Describe why the categorization might be useful.

# Define your custom categories and their summaries
custom_categories = {
    'Work Ethic and Discipline': (('Disciplined', 'Not Disciplined'),
                                  "This category assesses whether employees follow company norms related to punctuality, reliability, and work ethic."),

    'Potential vs. Performance': (('High Potential, Low Performance', 'Low Potential, Low Performance'),
                                  "This category captures the distinction between employees who have untapped potential versus those who are underperforming without much potential for growth.")
}

# Create a dictionary that maps (Category, Label) -> Example
example_dict = {
    ('Work Ethic and Discipline', 'Disciplined'): "Always on time and completes tasks promptly.",
    ('Work Ethic and Discipline', 'Not Disciplined'): "Takes many breaks, arrives late, and leaves early.",
    ('Potential vs. Performance', 'High Potential, Low Performance'): "Nice person, but their current work output is disappointing. Shows potential but needs guidance.",
    ('Potential vs. Performance', 'Low Potential, Low Performance'): "Shows no promise and consistently fails to meet standards."
}

# Display the categories and examples
for category in custom_categories:
    print(f"Category is: {category}; {custom_categories[category][1]}")
    print(f"Labels are: {list(enumerate(custom_categories[category][0]))}")
    print(f"Example for 'Disciplined' in '{category}': {example_dict[(category, 'Disciplined')]}") if ('Disciplined') in custom_categories[category][0] else None
    print()


#### In 3-5 sentences, explain why each category might be useful

**The Work Ethic and Discipline category is useful because it helps identify employees who struggle with punctuality, reliability, or adherence to workplace norms. This distinction allows managers to understand whether poor performance is due to a lack of effort or other factors, guiding decisions about support or disciplinary actions.**

**The Potential vs. Performance category is valuable because it highlights the difference between employees who underperform but have the potential to improve and those who show no promise for future growth. This categorization can help in tailoring interventions, such as training for high-potential employees, or making decisions about retention for low-potential individuals.**

Now we can add a new column to the dataset for each category. We will use -1 to denote unlabeled entries and the order of the labels in the dictionaries above to map numbers to labels.

# Add new columns for each category with an initial value of -1
for category in custom_categories:
    df_train[category] = -1
    df_test[category] = -1

# Function to label each row based on user input
def custom_label(row, category, labels, summary):
    """
    This function prompts the user to assign a label to each row for the given category.
    """
    print(f"Feedback is: {row['feedback']}")
    print(f"Original Label: {labels_dict_reversed[row['label']]}")
    print(f"Summary for category is: {summary}")

    # Display options and ask the user for input
    print(f"Labels: {list(enumerate(labels))}")
    label = input(f"Enter the index of the correct label for {category}: ")

    # Return the index of the chosen label
    return int(label)

# Function to label each category for a specific number of examples
def label_category(category, labels):
    """
    This function iterates through the training and test datasets, prompting the user to assign a label
    for each example in the given category.
    """
    print(f"Category is: {category}; {custom_categories[category][1]}")
    print(f"Labels are: {list(enumerate(labels))}")

    # Print an example for each label
    for label in labels:
        print(f"Example for {label} is: {example_dict[(category, label)]}")

    print('\n=======================================================\n')

    # Label training set
    for i in range(len(df_train)):
        # Label the training data for this category
        df_train.iloc[i, df_train.columns.get_loc(category)] = custom_label(
            df_train.iloc[i], category, labels, custom_categories[category][1]
        )

        # Label the testing data for this category (optional)
        df_test.iloc[i, df_test.columns.get_loc(category)] = custom_label(
            df_test.iloc[i], category, labels, custom_categories[category][1]
        )

        # Stop after labeling the specified number of examples
        if i >= examples_to_inspect:
            break

        # Print the current iteration index
        if i % 3 == 0:
            print('=======================================================')
            print(f"========= Current i is {i} =========")

# Label each category in the custom categories
for category in custom_categories:
    print('=======================================================')
    label_category(category, labels=custom_categories[category][0])
    print('=======================================================')


Print the df_train head and see the additional columns

print(df_train.head())

## Question 6: Building a Model for Your Labels

Now that we have some labels here, we will experiment with some methods for predicting your custom labels. One way to approach this would be to use the same method we did up above. If you'd like, you can implement that method and evaluate the new model quantiatively. However, you will likely find that performance is quite poor --- we tend to need more labels than is educationally productive to assign in a PSET!

An alternative is to turn our classification problem into a text completion task. In practice, this involves phrasing the classification task in natural language (i.e., English for our purposes here). For instance, if we wanted to classify whether sentence "I'm really disappointed that summer is over." is happy or sad, we could ask a language model to complete the following sentence:


```
"""
Is the following sentence happy?. Answer with "happy", "sad", or "neither". "I'm really disappointed that summer is over."
"""
```

While it isn't gauranteed that this will work, in practice these types of methods can work well with 0 additional data --- as long as they are evaluated properly. While this might seem like magic (and it is amazing), it's always a good idea to be skeptical of what seems like a free lunch. There are some important caveats that we need to consider:

*   The way you phrase your question directly determines the categories the model will recognize. Everything else will be influenced by the model's inherent biases. This isn't necessarily bad (for example, the model's bias towards generating coherent responses is crucial), but it means the model will rely on its data-driven interpretation of your prompt, which may or may not align with your intended meaning.

*  Small changes to prompts can change lead to large changes in behavior. [For example](https://arxiv.org/abs/2309.03409), including the phrase "Take a deep breath and work on this problem step-by-step" or offering to "pay" the model for better answers has improved performance in experiments. The process of iterating on these *prompts* has gotten the name *prompt engineering*. It's more of an iterative design exercise then a science, but [best practices](https://llama.meta.com/docs/how-to-guides/prompting/) are starting to emerge.

#### Optional Reading
While the idea of reducing one problem to another that you already know how to solve is one of the fundamentals in computer science, the application of this to large language models doing text completion tasks was introduced in "[Language Models are Unsupervised Multitask Learners.](https://paperswithcode.com/paper/language-models-are-unsupervised-multitask)"

### Question 6.1

Write out 3 potential prompts for each custom category you created. Explain your reasoning for the prompt in 1-3 sentences each. The prompts should be phrased as a question that the model can complete, aim to classify between the classes you created.

For example, if you had a category called "Work Quality" with labels "bad", "moderately bad", "moderately good" and "good worker", an example of a prompt can be: "Please classify the work quality of this employee according to the next list- bad, moderately bad, moderately good or good worker: {}"
and where the {} is replaced by the sentence you want to classify, which in our case is the feedback field.

prompts = {
    'Work Ethic and Discipline': [
        'Does this employee show good discipline at work? Answer with "Disciplined" or "Not Disciplined". Feedback: {}',
        ## WHY MIGHT THIS PROMPT WORK?
        'This prompt clearly directs the model to assess work discipline with a binary choice, focusing on whether the employee is disciplined or not.',

        'Based on the following feedback, how would you rate the employee\'s work ethic? Choose between "Disciplined" and "Not Disciplined". Feedback: {}',
        ## WHY MIGHT THIS PROMPT WORK?
        'This prompt highlights work ethic, ensuring the model considers employee behavior in terms of reliability and discipline.',

        'How reliable is this employee? Provide one of these answers: "Disciplined" or "Not Disciplined". Feedback: {}',
        ## WHY MIGHT THIS PROMPT WORK?
        'Asking about reliability prompts the model to focus on punctuality and dependability, key aspects of discipline.'
    ],

    'Potential vs. Performance': [
        'Does this employee have potential to grow in the company, or are they underperforming? Choose between "High Potential, Low Performance" or "Low Potential, Low Performance". Feedback: {}',
        ## WHY MIGHT THIS PROMPT WORK?
        'This prompt asks the model to explicitly assess both potential and current performance, making it easy to differentiate between the two labels.',

        'Based on the feedback, is this employee underperforming with potential to improve, or showing little potential for growth? Choose "High Potential, Low Performance" or "Low Potential, Low Performance". Feedback: {}',
        ## WHY MIGHT THIS PROMPT WORK?
        'This prompt breaks down the decision into two parts, helping the model assess both underperformance and potential clearly.',

        'How would you classify the employee\'s current performance and future potential? Choose between "High Potential, Low Performance" and "Low Potential, Low Performance". Feedback: {}',
        ## WHY MIGHT THIS PROMPT WORK?
        'The prompt encourages the model to assess both current performance and future potential, prompting a more balanced evaluation.'
    ]
}


Now that we have our prompts, we can evaluate them! We'll use the DistilBART language model, applying the designed prompts to the feedback field from the employees.

For example, consider an interaction with the model. Suppose the prompt you designed is: 'Does the following employee have good potential? Answer with "no", "mild potential", or "extraordinary potential",' and the feedback field is: 'John's performance was poor in the last quarter, although he graduated first in his class at MIT and scored an A+ in the AI, Decision-Making, and Society course.' In this case, we expect the model to return 'extraordinary potential'.

from transformers import pipeline
import torch
device = 0 if torch.cuda.is_available() else -1

# To accomplish that task, we will use the Bart model
classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-1", device = device)

# Function to classify feedback based on prompt
def classify_feedback(feedback, prompt, labels):
    # Combine prompt with feedback
    combined_input = prompt.format(feedback)

    # Perform zero-shot classification
    classification_result = classifier(combined_input, candidate_labels=labels)

    # Return the label with the highest score
    return classification_result['labels'][0]

# Apply summarization prompt
# Apply the prompts to each feedback and store the generated results
number_to_classify = 5
df_train['category_0_prompt_0'] = df_train.head(number_to_classify)['feedback'].apply(lambda feedback: classify_feedback(feedback, prompts[list(custom_categories.keys())[0]][0], custom_categories[list(custom_categories.keys())[0]][0]))
df_train['category_0_prompt_1'] = df_train.head(number_to_classify)['feedback'].apply(lambda feedback: classify_feedback(feedback, prompts[list(custom_categories.keys())[0]][1], custom_categories[list(custom_categories.keys())[0]][0]))
df_train['category_0_prompt_2'] = df_train.head(number_to_classify)['feedback'].apply(lambda feedback: classify_feedback(feedback, prompts[list(custom_categories.keys())[0]][2], custom_categories[list(custom_categories.keys())[0]][0]))

df_train['category_1_prompt_0'] = df_train.head(number_to_classify)['feedback'].apply(lambda feedback: classify_feedback(feedback, prompts[list(custom_categories.keys())[1]][0], custom_categories[list(custom_categories.keys())[1]][0]))
df_train['category_1_prompt_1'] = df_train.head(number_to_classify)['feedback'].apply(lambda feedback: classify_feedback(feedback, prompts[list(custom_categories.keys())[1]][1], custom_categories[list(custom_categories.keys())[1]][0]))
df_train['category_1_prompt_2'] = df_train.head(number_to_classify)['feedback'].apply(lambda feedback: classify_feedback(feedback, prompts[list(custom_categories.keys())[1]][2], custom_categories[list(custom_categories.keys())[1]][0]))

# print the results
print(df_train[['feedback', list(custom_categories.keys())[0], 'category_0_prompt_0']].head(number_to_classify))
print(df_train[['feedback', list(custom_categories.keys())[0], 'category_0_prompt_1']].head(number_to_classify))
print(df_train[['feedback', list(custom_categories.keys())[0], 'category_0_prompt_2']].head(number_to_classify))

print(df_train[['feedback', list(custom_categories.keys())[1], 'category_1_prompt_0']].head(number_to_classify))
print(df_train[['feedback', list(custom_categories.keys())[1], 'category_1_prompt_1']].head(number_to_classify))
print(df_train[['feedback', list(custom_categories.keys())[1], 'category_1_prompt_2']].head(number_to_classify))

### Question 6.2 - Few-Shot Prompting [Only for students enrolled in the graduate version of the class]

We now aim to perform few-shot prompting, where we use our zero-shot classifier (from the previous question) to create prompts that guide the model to classify unseen labels. Specifically, the model will use a few labeled examples in the prompt to enhance its understanding of new labels. Additionally, we would like to implement a function that re-generates prompts based on the predictions made by the model on the previous prompts, incorporating them into future examples to improve the prompting mechanism iteratively.

Given the previous prompts and the model's predictions, check the response of the model on some new unseen prompts and labels. You can use the previous prompts and labels to generate the new prompts and labels, thus "guiding" the model towards the right answer by giving him the previously generated information.

# Function to classify feedback without few-shot examples for debugging purposes
def classify_feedback_simple(feedback, prompt, labels):
    # Combine the prompt with the feedback (no few-shot examples for now)
    combined_input = prompt.format(feedback)

    # Debug: Print combined input to ensure it looks correct
    print(f"\n[DEBUG] Combined Input:\n{combined_input}")

    # Perform zero-shot classification using the existing classifier pipeline
    classification_result = classifier(combined_input, candidate_labels=labels)

    # Debug: Print the classification result to check its structure
    print(f"\n[DEBUG] Classification Result:\n{classification_result}")

    # Return the label with the highest score (check if the output contains valid labels)
    if 'labels' in classification_result and classification_result['labels']:
        return classification_result['labels'][0]
    else:
        # Debug: If the classifier doesn't return valid labels, return 'Unknown'
        print("\n[DEBUG] No valid labels found. Returning 'Unknown'.\n")
        return 'Unknown'

# Test the simpler process with just the feedback and prompt (no few-shot examples)
number_to_classify = 5

# Iterate over the feedback and apply zero-shot classification (no few-shot prompting)
for i in range(number_to_classify):
    # Classify the feedback using simplified process for Category 0
    new_label_0 = classify_feedback_simple(
        df_train['feedback'].iloc[i],
        prompts[list(custom_categories.keys())[0]][0],  # Using the first category and first prompt
        custom_categories[list(custom_categories.keys())[0]][0]  # Labels for the first category
    )

    # Classify the feedback using simplified process for Category 1
    new_label_1 = classify_feedback_simple(
        df_train['feedback'].iloc[i],
        prompts[list(custom_categories.keys())[1]][0],  # Using the second category and first prompt
        custom_categories[list(custom_categories.keys())[1]][0]  # Labels for the second category
    )

    # Store the results in the dataframe
    df_train.at[i, 'category_0_simple'] = new_label_0
    df_train.at[i, 'category_1_simple'] = new_label_1

# Present the results in a more organized format
print("\nSimplified Classification Results\n")
print("=" * 40)

for i in range(number_to_classify):
    print(f"\nFeedback {i+1}:")
    print(f"---------------------")
    print(f"{df_train['feedback'].iloc[i]}")

    print("\nCategory 0 (Work Ethic and Discipline):")
    print(f"  - Prediction: {df_train['category_0_simple'].iloc[i]}")

    print("\nCategory 1 (Potential vs. Performance):")
    print(f"  - Prediction: {df_train['category_1_simple'].iloc[i]}")

    print("\n" + "=" * 40)

## Question 7: Reflections on the overall process

1. In your own words, describe the steps we used to build and evaluate models for employee review. What additional steps would you recommend for a company thinking of deploying such a model?

2. How did your re-labeling change the performance? What does this indicate about the "quality" of the initial labels?

3. If a company were to deploy such a model for employment review purposes, what are some pros and cons of using your custom categorization of the data compared to the original labels?

4. What is the key idea behind using a language model for "zero-shot classification"? Did certain prompts work better than others, and why do you think that was the case?

1. In your own words, describe the steps we used to build and evaluate models for employee review. What additional steps would you recommend for a company thinking of deploying such a model?

**The process involved several key steps. First, we loaded the employee review dataset and performed exploratory data analysis to understand the structure. Next, we used a zero-shot classification model to categorize the feedback based on custom categories derived from manual annotations. To refine the labels, we implemented few-shot learning by incorporating examples from previous predictions into new prompts to improve the model's performance iteratively. Additional steps for a company deploying such a model would include expanding the labeled dataset for better training, conducting thorough bias assessments, and testing the model under real-world conditions to ensure reliability and fairness in performance reviews.**

2. How did your re-labeling change the performance? What does this indicate about the "quality" of the initial labels?

**Re-labeling improved the performance of the model by providing more consistent and accurate labels compared to the original ones. This change suggests that the initial labels in the dataset were either noisy or inconsistently applied, impacting model accuracy. The re-labeling process highlighted the importance of having high-quality, well-defined labels, which directly influence model performance. It also indicates that more rigorous data labeling could lead to better model outcomes.**

3. If a company were to deploy such a model for employment review purposes, what are some pros and cons of using your custom categorization of the data compared to the original labels?

**Using custom categorization allows for a more tailored analysis of employee feedback, focusing on specific behaviors or characteristics that may be more relevant to a company’s culture. Pros include more relevant insights and the ability to address specific performance aspects like work ethic or potential. However, the cons include increased complexity in categorizing the data, which may lead to bias if categories are too subjective. Additionally, there could be inconsistencies in label interpretation between human annotators, potentially leading to inaccurate model predictions.**

4. What is the key idea behind using a language model for "zero-shot classification"? Did certain prompts work better than others, and why do you think that was the case?

**The key idea behind zero-shot classification is that the language model can generalize and classify unseen data based on a provided prompt without requiring training on labeled examples for that specific task. Certain prompts worked better than others because the language model’s performance relies heavily on how well the task is framed within the prompt. Simpler and more direct prompts often resulted in better predictions because they reduced ambiguity and helped the model align its output with the intended classification task. This indicates that prompt clarity and structure are crucial for the success of zero-shot classification.**

### For graduate students

5. How did few-shot prompting change the performance? What strategies did you find useful for choosing the few-shot examples?  

**Few-shot prompting improved the model's performance by providing context through labeled examples, allowing it to better understand new, unseen feedback. Useful strategies for choosing few-shot examples included selecting diverse and representative examples from different categories, as this helped the model generalize better. Additionally, using clear and consistent feedback-label pairs as few-shot examples helped guide the model toward more accurate predictions.**
