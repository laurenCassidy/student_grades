# Student Performance Prediction

This project aims to predict the final grade (G3) of students based on their second-semester grade (G2). It utilizes a linear regression model for prediction.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [License](#license)

## Prerequisites

To run this project, you need to have the following libraries and tools installed:

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Getting Started

1. Clone this repository to your local machine.

   ```bash
   git clone https://github.com/yourusername/student-performance-prediction.git
   ```

2. Change to the project directory.

   ```bash
   cd student-performance-prediction
   ```

3. Download the dataset (student-mat.csv) and place it in the project directory.

4. Run the Jupyter notebook to execute the code.

## Usage

The primary purpose of this project is to predict the final grade (G3) of students based on their second-semester grade (G2). The linear regression model is trained using the provided dataset (student-mat.csv). The model is then saved to a file (studentmodel.pickle) for later use.

To make predictions using the trained model, you can load the model from the pickle file and use it to predict the final grade (G3) for a given second-semester grade (G2).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
