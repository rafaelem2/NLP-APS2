# Supervised Practice - APS 2: Vector-Based Search

## Dataset Description

The dataset consists of profiles of 290 athletes, with each entry providing detailed information about individual achievements, biographical highlights, and significant career milestones. For each athlete, the dataset includes the following attributes:

- **Name**: The athlete's full name.
- **MainContent**: General profile information, including nationality and sport.
- **BioMedals**: Information about the events the athlete participated in and any medals won.
- **BiographicalInformation**: Additional personal information, including achievements and career highlights.
- **Milestones**: Notable moments and achievements in the athlete's career.

This dataset is well-suited for an embedding-based retrieval system, enabling semantic search capabilities based on similarities in achievements, biographical details, and career milestones. Its structured information offers a rich context for identifying relationships between athletes in terms of their sports, accomplishments, and historical significance.

### Embedding Generation Process

We generated embeddings for each athlete profile using **SBERT (paraphrase-MiniLM-L6-v2)**, a model known for capturing nuanced semantic relationships at the sentence level. To further adapt these embeddings to our dataset, we applied a denoising autoencoder with an **input layer of 384 dimensions** (SBERT embedding size), a **hidden layer of 128 dimensions** with ReLU activation for dimensionality reduction, and a **decoder layer** to reconstruct the embeddings back to 384 dimensions. This autoencoder was trained using **Mean Squared Error (MSE)** as the loss function, with the **Adam optimizer** (learning rate = 0.001) over **10 epochs** to minimize reconstruction error. This process refined the embeddings to better represent our dataset, enhancing the quality of our search system.


### Training Process

To fine-tune the SBERT embeddings, we trained a denoising autoencoder that reconstructs the original embeddings after reducing their dimensionality. The training process minimizes reconstruction error using **Mean Squared Error (MSE)** as the loss function, which calculates the average of the squared differences between the original embeddings and their reconstructed versions. This loss function is suitable because it effectively preserves the core semantic features of the embeddings while allowing for dimensionality reduction. By minimizing MSE, the model retains important information in the lower-dimensional space, making the embeddings more relevant to the specific athlete profiles in our dataset.

The MSE loss function is defined as:

\[
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (x_i - \hat{x}_i)^2
\]

where \( x_i \) represents each original embedding feature, \( \hat{x}_i \) is the corresponding reconstructed feature, and \( n \) is the total number of features.
