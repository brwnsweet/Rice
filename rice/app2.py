import tkinter as tk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from ucimlrepo import fetch_ucirepo

# fetch dataset 
rice_cammeo_and_osmancik = fetch_ucirepo(id=545)
# data (as pandas dataframes) 
X = rice_cammeo_and_osmancik.data.features 
y = rice_cammeo_and_osmancik.data.targets 

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train KNN classifier
k = 3  # K value
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_classifier.fit(X_train, y_train)

# Function to predict flower species
def predict_species():
    try:
        # Get input values
        Perimeter = float(entry_Perimeter.get())
        Major_Axis_Length = float(entry_Major_Axis_Length.get())
        Minor_Axis_Length = float(entry_Minor_Axis_Length.get())
        Eccentricity = float(entry_Eccentricity.get())
        
        
        # Standardize input
        input_data = scaler.transform([[Perimeter, Major_Axis_Length, Minor_Axis_Length, Eccentricity,]])
        
        # Predict
        prediction = knn_classifier.predict(input_data)
        
        # Map prediction to rice species
        species = prediction.target_names[prediction[0]]
        
        # Show prediction
        messagebox.showinfo("Prediction", f"The predicted species is: {species}")
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numerical values for all input fields.")

# Create GUI
root = tk.Tk()
root.title("Rice Species Prediction")

# Labels
label_Perimeter = tk.Label(root, text="Perimeter:")
label_Perimeter.grid(row=0, column=0)
label_Major_Axis_Length = tk.Label(root, text="Major Axis Length:")
label_Major_Axis_Length.grid(row=1, column=0)
label_Minor_Axis_Length = tk.Label(root, text="Minor Axis Length:")
label_Minor_Axis_Length.grid(row=2, column=0)
label_Eccentricity = tk.Label(root, text="Eccentricity:")
label_Eccentricity.grid(row=3, column=0)

# Entry fields
entry_Perimeter = tk.Entry(root)
entry_Perimeter.grid(row=0, column=1)
entry_Major_Axis_Length = tk.Entry(root)
entry_Major_Axis_Length.grid(row=1, column=1)
entry_Minor_Axis_Length = tk.Entry(root)
entry_Minor_Axis_Length.grid(row=2, column=1)
entry_Eccentricity = tk.Entry(root)
entry_Eccentricity.grid(row=3, column=1)

# Button
button_predict = tk.Button(root, text="Predict", command=predict_species)
button_predict.grid(row=4, column=0, columnspan=2)

root.mainloop()

#=============
#nadia erica
#nuril izza
#=============