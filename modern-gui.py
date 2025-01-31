import pandas as pd
from tkinter import *
from tkinter import ttk, messagebox
import tkinter.font as tkFont
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from ttkthemes import ThemedTk

class ModernTyreRecommender:
    def __init__(self):
        # Load and preprocess data
        self.load_data()
        
        # Create themed root window
        self.root = ThemedTk(theme="arc")
        self.root.title("Tire Recommendation System")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')
        
        # Configure styles
        self.style = ttk.Style()
        self.style.configure('Title.TLabel', font=('Helvetica', 24, 'bold'), padding=20)
        self.style.configure('Header.TLabel', font=('Helvetica', 12, 'bold'), padding=5)
        self.style.configure('Custom.TCombobox', padding=5)
        self.style.configure('Action.TButton', font=('Helvetica', 12), padding=10)
        
        self.create_widgets()

    def load_data(self):
        # Load the dataset
        self.data = pd.read_csv("Car_Tyres_Dataset.csv")
        self.data.columns = self.data.columns.str.strip()
        print("Column names after cleaning:", list(self.data.columns))

        # Preprocess the dataset
        self.data["Rating"] = self.data["Rating"].fillna(self.data["Rating"].mean())
        self.data["Original Price"] = self.data["Original Price"].fillna(self.data["Selling Price"])

        # Encode categorical data
        self.label_encoders = {}
        for column in ["Brand", "Model", "Submodel", "Tyre Brand", "Type", "Size"]:
            le = LabelEncoder()
            self.data[column] = le.fit_transform(self.data[column].astype(str))
            self.label_encoders[column] = le

        # Train the recommendation model
        self.X = self.data[["Brand", "Model", "Submodel", "Type", "Size"]]
        self.knn = NearestNeighbors(n_neighbors=5, metric="euclidean")
        self.knn.fit(self.X)
        
    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=BOTH, expand=True)
        
        # Title
        title = ttk.Label(main_frame, text="Tire Recommendation System", style='Title.TLabel')
        title.pack(pady=(0, 20))
        
        # Create form frame
        form_frame = ttk.Frame(main_frame)
        form_frame.pack(fill=BOTH, padx=50)
        
        # Variables
        self.brand_var = StringVar()
        self.model_var = StringVar()
        self.submodel_var = StringVar()
        self.type_var = StringVar(value="Tubeless")
        self.tyre_size_var = StringVar()
        
        # Brand selection
        brand_values = list(self.label_encoders["Brand"].classes_)
        self.create_form_field(form_frame, "Vehicle Brand:", self.brand_var, 
                             brand_values, 0, self.update_models)
        
        # Model selection
        self.create_form_field(form_frame, "Vehicle Model:", self.model_var, 
                             [], 1, self.update_submodels)
        
        # Submodel selection
        self.create_form_field(form_frame, "Vehicle Submodel:", self.submodel_var, 
                             [], 2, self.update_expected_tyre_size)
        
        # Tire type selection
        self.create_form_field(form_frame, "Tire Type:", self.type_var, 
                             ["Tubeless", "Tube"], 3)
        
        # Tire size field
        size_label = ttk.Label(form_frame, text="Expected Tire Size:", style='Header.TLabel')
        size_label.grid(row=4, column=0, sticky=W, pady=5)
        
        size_entry = ttk.Entry(form_frame, textvariable=self.tyre_size_var, width=30)
        size_entry.grid(row=4, column=1, sticky=W, pady=5)
        
        # Recommendation button
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=X, pady=30)
        
        recommend_btn = ttk.Button(button_frame, text="Get Recommendations", 
                                 style='Action.TButton', command=self.recommend)
        recommend_btn.pack(pady=10)
        
        # Results frame
        self.results_frame = ttk.Frame(main_frame)
        self.results_frame.pack(fill=BOTH, expand=True)

    def create_form_field(self, parent, label_text, variable, values, row, trace_command=None):
        label = ttk.Label(parent, text=label_text, style='Header.TLabel')
        label.grid(row=row, column=0, sticky=W, pady=5)
        
        combo = ttk.Combobox(parent, textvariable=variable, values=values, 
                            state='readonly', width=27, style='Custom.TCombobox')
        combo.grid(row=row, column=1, sticky=W, pady=5)
        
        if trace_command:
            variable.trace("w", trace_command)

    def update_models(self, *args):
        selected_brand = self.brand_var.get()
        brand_idx = self.label_encoders["Brand"].transform([selected_brand])[0]
        models = self.data[self.data["Brand"] == brand_idx]["Model"].unique()
        model_names = self.label_encoders["Model"].inverse_transform(models)
        self.update_combobox_values(self.model_var, model_names)
        self.model_var.set("")
        self.submodel_var.set("")
        self.tyre_size_var.set("")

    def update_submodels(self, *args):
        if self.model_var.get():
            model_idx = self.label_encoders["Model"].transform([self.model_var.get()])[0]
            submodels = self.data[self.data["Model"] == model_idx]["Submodel"].unique()
            submodel_names = self.label_encoders["Submodel"].inverse_transform(submodels)
            self.update_combobox_values(self.submodel_var, submodel_names)
            self.submodel_var.set("")
            self.tyre_size_var.set("")

    def update_expected_tyre_size(self, *args):
        if self.submodel_var.get():
            submodel_idx = self.label_encoders["Submodel"].transform([self.submodel_var.get()])[0]
            size_idx = self.data[self.data["Submodel"] == submodel_idx]["Size"].mode()[0]
            self.tyre_size_var.set(self.label_encoders["Size"].inverse_transform([size_idx])[0])

    def update_combobox_values(self, variable, values):
        for widget in self.root.winfo_children():
            if isinstance(widget, ttk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, ttk.Frame):
                        for grandchild in child.winfo_children():
                            if isinstance(grandchild, ttk.Combobox) and str(grandchild.cget("textvariable")) == str(variable):
                                grandchild['values'] = sorted(values)

    def recommend(self):
        try:
            # Create a mapping of column names to variable names
            var_mapping = {
                "Brand": "brand_var",
                "Model": "model_var",
                "Submodel": "submodel_var",
                "Type": "type_var",
                "Size": "tyre_size_var"  # Changed from size_var to tyre_size_var
            }
            
            # Get input values using the correct variable names
            input_vector = [
                self.label_encoders[col].transform([getattr(self, var_mapping[col]).get()])[0]
                for col in ["Brand", "Model", "Submodel", "Type", "Size"]
            ]
            
            distances, indices = self.knn.kneighbors([input_vector])
            recommendations = self.data.iloc[indices[0]].copy()
            
            # Decode the categorical values back to their original form
            recommendations["Tyre Brand"] = self.label_encoders["Tyre Brand"].inverse_transform(recommendations["Tyre Brand"])
            recommendations["Size"] = self.label_encoders["Size"].inverse_transform(recommendations["Size"])
            
            # Display only the relevant columns
            self.display_recommendations(recommendations[["Tyre Brand", "Size", "Selling Price", "Original Price", "Rating"]])
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            print(f"Debug info - Error details: {e}")  # Added for debugging

    def display_recommendations(self, recommendations):
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()
            
        # Create Treeview
        columns = ("Tyre Brand", "Size", "Selling Price", "Original Price", "Rating")
        tree = ttk.Treeview(self.results_frame, columns=columns, show='headings')
        
        # Configure columns
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=120)
        
        # Add data
        for _, row in recommendations.iterrows():
            values = [row[col] for col in columns]
            tree.insert('', END, values=values)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(self.results_frame, orient=VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack widgets
        tree.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)

if __name__ == "__main__":
    app = ModernTyreRecommender()
    app.root.mainloop()