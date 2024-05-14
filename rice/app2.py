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
        Area = float(entry_Area.get())
        Perimeter = float(entry_Perimeter.get())
        Major_Axis_Length = float(entry_Major_Axis_Length.get())
        Minor_Axis_Length = float(entry_Minor_Axis_Length.get())
        Eccentricity = float(entry_Eccentricity.get())
        Convex_Area = float(entry_Convex_Area.get())
        Extent = float(entry_Extent.get())
        
        
        # Standardize input
        input_data = scaler.transform([[Area, Perimeter, Major_Axis_Length, Minor_Axis_Length, Eccentricity, Convex_Area, Extent]])
        
        # Predict
        prediction = knn_classifier.predict(input_data)
             
        # Show prediction
        messagebox.showinfo("Prediction", f"The predicted species is: {prediction}")
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numerical values for all input fields.")

# Create GUI 
root = tk.Tk()
root.title("Rice Species Prediction")

# Labels
label_Area = tk.Label(root, text="Area:")
label_Area.grid(row=0, column=0)
label_Perimeter = tk.Label(root, text="Perimeter:")
label_Perimeter.grid(row=1, column=0)
label_Major_Axis_Length = tk.Label(root, text="Major Axis Length:")
label_Major_Axis_Length.grid(row=2, column=0)
label_Minor_Axis_Length = tk.Label(root, text="Minor Axis Length:")
label_Minor_Axis_Length.grid(row=3, column=0)
label_Eccentricity = tk.Label(root, text="Eccentricity:")
label_Eccentricity.grid(row=4, column=0)
label_Convex_Area = tk.Label(root, text="Convex Area:")
label_Convex_Area.grid(row=5, column=0)
label_Extent = tk.Label(root, text="Extent:")
label_Extent.grid(row=6, column=0)

# Entry fields
entry_Area = tk.Entry(root)
entry_Area.grid(row=0, column=1)
entry_Perimeter = tk.Entry(root)
entry_Perimeter.grid(row=1, column=1)
entry_Major_Axis_Length = tk.Entry(root)
entry_Major_Axis_Length.grid(row=2, column=1)
entry_Minor_Axis_Length = tk.Entry(root)
entry_Minor_Axis_Length.grid(row=3, column=1)
entry_Eccentricity = tk.Entry(root)
entry_Eccentricity.grid(row=4, column=1)
entry_Convex_Area = tk.Entry(root)
entry_Convex_Area.grid(row=5, column=1)
entry_Extent = tk.Entry(root)
entry_Extent.grid(row=6, column=1)

# Button
button_predict = tk.Button(root, text="Predict", command=predict_species)
button_predict.grid(row=8, column=0, columnspan=2)

root.mainloop()

#=============
#nadia erica
#nuril izza
#=============