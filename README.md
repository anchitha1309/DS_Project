# Rice Leaf Disease Detection Using CNN

## Abstract
This project endeavors to tackle the critical issue of rice leaf diseases through advanced deep learning methodologies. By developing a sophisticated convolutional neural network (CNN) model and an intuitive Streamlit web application, this endeavor aims to provide farmers and agricultural experts with a reliable tool for accurate disease classification and diagnosis.

## Dataset
Access to a comprehensive dataset comprising labeled rice leaf images is imperative for model training and evaluation. Ensure meticulous organization and preprocessing of the dataset into distinct training and validation subsets to facilitate effective model learning and validation.
[Dataset Link](https://www.kaggle.com/datasets/isaacritharson/severity-based-rice-leaf-diseases-dataset)

### Class Labels

The model is trained to classify rice leaf images into the following nine classes:

1. ⁠**⁠Healthy**
2. ⁠**⁠Mild Bacterial blight**
3. ⁠**⁠Mild Blast**
4. ⁠**⁠Mild Brownspot**
5. ⁠**⁠Mild Tungro**
6. ⁠**⁠Severe Bacterial blight**
7. ⁠**⁠Severe Blast**
8. ⁠**⁠Severe Brownspot**
9. ⁠**⁠Severe Tungro**

## Requirements
- Python 3.x
- TensorFlow
- Keras
- Streamlit
- Matplotlib
- Numpy
- Scikit-learn

## Model Development

### AI_Project.ipynb
The AI_Project.ipynb notebook serves as the cornerstone of this project, encapsulating the meticulous process of constructing and training the CNN model. Key highlights of this notebook include:

- **Data Preprocessing**: Leveraging the powerful ImageDataGenerator for meticulous data augmentation and preprocessing tasks to enhance model robustness.
- **Model Architecture Definition**: Crafting a state-of-the-art CNN architecture comprising convolutional layers, batch normalization, activation functions, max-pooling layers, and dense layers for effective feature extraction and classification.
- **Model Compilation and Training**: Employing the RMSProp optimizer and categorical crossentropy loss function for model compilation, coupled with strategic learning rate scheduling to ensure optimal convergence during training.
- **Model Evaluation**: Rigorous evaluation of model performance on validation data to validate efficacy and identify potential areas for improvement.
- **Model Preservation**: Preservation of the trained model as my_model_rmsprop_with_bn.h5, ensuring accessibility and reusability for future applications.

## Streamlit Visualization

### leaf_stram_visu.py
The leaf_stram_visu.py script represents the culmination of efforts to democratize access to advanced model functionalities through an intuitive Streamlit web application. Notable features of this script include:

- **Real-time Inference**: Seamless integration with the trained model to enable real-time inference on uploaded images, providing instant feedback to users.
- **Interactive Visualization**: Intuitive sliders and dynamic visualization components to empower users to explore and interpret intermediate activations of the model, fostering deeper understanding.
- **User-friendly Interface**: Clean and intuitive user interface design with clear instructions and visual cues for enhanced usability and accessibility.
- **Validation and Feedback**: Instantaneous display of predicted classes and uploaded images, enabling users to validate model predictions and provide feedback for continuous improvement.

## Usage Instructions

1. **Model Training**: Execute the AI_Project.ipynb notebook to initiate model training. Fine-tune hyperparameters and architectural components as necessary to achieve desired performance benchmarks.
2. **Model Deployment**: Utilize the leaf_stram_visu.py script to deploy the trained model([my_model (1).h5](https://github.com/anchitha1309/Rice_Leaf_Disease-Detection_streamlit/blob/main/my_model%20(1).h5)) via the Streamlit web application. Ensure seamless integration and adherence to all dependencies and configurations.
3. **User Interaction**: Encourage users to interact with the Streamlit web application to upload images, visualize intermediate activations, and validate predictions. Foster an engaging and informative user experience to drive adoption and utilization.

## Model Evaluation

The trained model is evaluated on a separate test set, and the classification report is printed, showing the precision, recall, F1-score, and support for each class, as well as the overall accuracy.

To cite the relevant search results:

The project overview, Streamlit deployment, and class labels are explained based on the code in  [leaf_stram_visu.py](https://github.com/anchitha1309/Rice_Leaf_Disease-Detection_streamlit/blob/main/leaf_stram_visu.py)⁠. The model architecture and training process are described using the information from ⁠ [AI_Project.ipynb](https://github.com/anchitha1309/Rice_Leaf_Disease-Detection_streamlit/blob/main/AI_Project.ipynb) ⁠. The model evaluation is also performed and reported in ⁠ [AI_Project.ipynb](https://github.com/anchitha1309/Rice_Leaf_Disease-Detection_streamlit/blob/main/AI_Project.ipynb) .

## Evaluation Perspective
This project has been developed with a focus on depth, complexity, and real-world applicability, aligning with the evaluation criteria from an employer's perspective. By addressing the intricate challenges of rice leaf disease classification and leveraging advanced deep learning techniques, this project showcases proficiency, competence, and a commitment to delivering impactful solutions.

## Use Case

### Agricultural Industry
- **Early Disease Detection**: The CNN model can accurately classify rice leaf diseases, enabling early detection of potential health issues in crops. By identifying diseases at an early stage, farmers can take timely preventive measures, such as targeted pesticide application or crop management practices, to mitigate crop losses and ensure optimal yield.
- **Precision Agriculture**: Integration of the CNN model into precision agriculture systems allows for precise and targeted interventions, reducing resource wastage and environmental impact. Farmers can optimize the use of fertilizers, pesticides, and water resources based on real-time disease information, leading to improved crop health and sustainability.
- **Decision Support System**: The Streamlit web application serves as a decision support system for farmers and agricultural experts. By visualizing intermediate activations and model predictions, users gain valuable insights into disease patterns, enabling informed decision-making regarding crop management strategies and resource allocation.

## How It Is Used in Companies

### AgTech Companies
- **Product Development**: AgTech companies can leverage the CNN model and Streamlit web application to develop innovative agricultural solutions focused on disease management and crop health monitoring. By incorporating machine learning algorithms into their products, companies can offer farmers advanced tools for optimizing agricultural practices and improving productivity.
- **Integration with IoT Devices**: Integration of the CNN model with IoT devices and sensors enables real-time monitoring of crop health and disease outbreaks. Companies can deploy sensor networks in fields to collect data on environmental conditions and plant health, feeding this data into the CNN model for analysis and decision-making.
- **Collaboration with Agricultural Stakeholders**: AgTech companies can collaborate with agricultural stakeholders, including farmers, agronomists, and research institutions, to deploy the CNN model and Streamlit web application in real-world agricultural settings. By fostering collaboration and knowledge sharing, companies can drive adoption and uptake of their technology solutions, ultimately contributing to sustainable agricultural practices and food security.

## References
- https://iopscience.iop.org/article/10.1088/1755-1315/1032/1/012017/pdf
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8049121/
- https://ieeexplore.ieee.org/document/9645888
