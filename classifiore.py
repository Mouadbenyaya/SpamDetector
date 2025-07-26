import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk, filedialog
import joblib
from datetime import datetime
import threading

class SpamDetector:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.is_trained = False
        self.training_accuracy = 0
        
    def clean_text(self, text):
        """Fonction de nettoyage de texte améliorée"""
        if pd.isna(text) or text is None:
            return ""
        
        text = str(text)
        
        # Suppression des caractères d'échappement spécifiques
        text = re.sub(r"escapenumber|escapelong", " ", text)
        
        # Suppression des URLs (plus robuste)
        text = re.sub(r"https?://\S+|www\.\S+|\S+\.com\S*|\S+\.net\S*|\S+\.org\S*", " ", text)
        
        # Suppression des emails
        text = re.sub(r"\S+@\S+", " ", text)
        
        # Suppression des numéros de téléphone et codes
        text = re.sub(r"\b\d{3,}\b", " ", text)
        
        # Suppression des caractères spéciaux mais garde les lettres accentuées
        text = re.sub(r"[^a-zA-ZÀ-ÿ\s]", " ", text)
        
        # Suppression des mots très courts (1-2 lettres) qui sont souvent du bruit
        text = re.sub(r"\b\w{1,2}\b", " ", text)
        
        # Conversion en minuscules
        text = text.lower()
        
        # Suppression des espaces multiples
        text = re.sub(r"\s+", " ", text).strip()
        
        return text

    def load_and_train_model(self, csv_path="data/combined_data.csv"):
        """Charge les données et entraîne le modèle"""
        try:
            # Vérification de l'existence du fichier
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Le fichier {csv_path} n'existe pas")
            
            # Chargement des données avec gestion d'erreurs
            try:
                # D'abord, essayer de lire avec header pour détecter s'il y en a un
                df_test = pd.read_csv(csv_path, nrows=1, encoding='utf-8')
                
                # Si la première ligne contient "label" et "text", c'est un header
                if 'label' in df_test.columns and 'text' in df_test.columns:
                    df = pd.read_csv(csv_path, encoding='utf-8')  # Avec header
                else:
                    df = pd.read_csv(csv_path, names=["label", "text"], encoding='utf-8')  # Sans header
                    
            except UnicodeDecodeError:
                # Essayer avec un autre encodage
                try:
                    df_test = pd.read_csv(csv_path, nrows=1, encoding='latin-1')
                    if 'label' in df_test.columns and 'text' in df_test.columns:
                        df = pd.read_csv(csv_path, encoding='latin-1')
                    else:
                        df = pd.read_csv(csv_path, names=["label", "text"], encoding='latin-1')
                except:
                    df = pd.read_csv(csv_path, names=["label", "text"], encoding='utf-8', header=None)
            
            print(f"Données chargées: {len(df)} lignes")
            print(f"Colonnes: {df.columns.tolist()}")
            print(f"Premières lignes:\n{df.head()}")
            
            # Vérification du format des données
            if len(df.columns) != 2:
                raise ValueError(f"Le fichier doit avoir exactement 2 colonnes (label, text), trouvé {len(df.columns)}")
            
            # Nettoyage des données
            df = df.dropna()  # Suppression des valeurs manquantes
            df = df[df['text'].str.len() > 0]  # Supprimer les textes vides
            
            # Vérification des labels
            unique_labels = df['label'].unique()
            print(f"Labels uniques: {unique_labels}")
            
            # S'assurer que les labels sont 0 et 1
            if not all(pd.api.types.is_numeric_dtype(type(label)) or label in [0, 1, '0', '1'] for label in unique_labels):
                print("Conversion des labels en format binaire...")
                # Si les labels ne sont pas numériques, les convertir
                if len(unique_labels) == 2:
                    label_mapping = {unique_labels[0]: 0, unique_labels[1]: 1}
                    df['label'] = df['label'].map(label_mapping)
                else:
                    raise ValueError(f"Nombre de labels incorrect: {len(unique_labels)}. Attendu: 2")
            
            # Convertir en int pour être sûr
            df['label'] = df['label'].astype(int)
            
            # Vérifier que les labels sont bien 0 et 1
            final_labels = df['label'].unique()
            if not all(label in [0, 1] for label in final_labels):
                raise ValueError(f"Labels invalides après conversion: {final_labels}")
            
            print(f"Distribution des labels: {df['label'].value_counts().to_dict()}")
            
            df["clean_text"] = df["text"].apply(self.clean_text)
            
            # Vérifier qu'il y a assez de données
            if len(df) < 10:
                raise ValueError(f"Pas assez de données pour l'entraînement: {len(df)} lignes")
            
            # Préparation des données
            X = df["clean_text"]
            y = df["label"]
            
            # Division train/test avec vérification
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            except ValueError as e:
                print(f"Erreur lors de la division train/test: {e}")
                # Si stratify échoue, essayer sans
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
            
            # Vectorisation avec paramètres plus conservateurs
            self.vectorizer = TfidfVectorizer(
                max_features=3000,  # Réduit pour éviter l'overfitting
                stop_words='english',
                ngram_range=(1, 1),  # Seulement unigrammes pour plus de stabilité
                min_df=5,  # Mots qui apparaissent au moins 5 fois
                max_df=0.8,  # Ignore les mots très fréquents
                lowercase=True,
                strip_accents='ascii'
            )
            
            X_train_vec = self.vectorizer.fit_transform(X_train)
            X_test_vec = self.vectorizer.transform(X_test)
            
            # Entraînement du modèle avec régularisation plus forte
            self.model = LogisticRegression(
                max_iter=1000,
                C=0.1,  # Régularisation plus forte pour éviter l'overfitting
                random_state=42,
                class_weight='balanced'  # Équilibrage automatique des classes
            )
            self.model.fit(X_train_vec, y_train)
            
            # Évaluation
            y_pred = self.model.predict(X_test_vec)
            self.training_accuracy = accuracy_score(y_test, y_pred)
            
            print(f"Précision du modèle : {self.training_accuracy:.4f}")
            print("\nRapport de classification :")
            print(classification_report(y_test, y_pred))
            
            self.is_trained = True
            return True, f"Modèle entraîné avec succès!\nPrécision: {self.training_accuracy:.2%}"
            
        except Exception as e:
            return False, f"Erreur lors de l'entraînement : {str(e)}"
    
    def predict_text(self, text):
        """Prédit si un texte est spam ou non avec diagnostic"""
        if not self.is_trained:
            return None, "Le modèle n'est pas encore entraîné"
        
        try:
            original_text = text[:200] + "..." if len(text) > 200 else text
            cleaned = self.clean_text(text)
            
            if not cleaned.strip():
                return None, "Le texte est vide après nettoyage"
            
            print(f"\n--- DIAGNOSTIC DE PRÉDICTION ---")
            print(f"Texte original (200 premiers chars): {original_text}")
            print(f"Texte nettoyé: {cleaned[:100]}...")
            
            vect = self.vectorizer.transform([cleaned])
            prediction = self.model.predict(vect)[0]
            probabilities = self.model.predict_proba(vect)[0]
            
            print(f"Probabilités: Normal={probabilities[0]:.3f}, Spam={probabilities[1]:.3f}")
            print(f"Prédiction: {'SPAM' if prediction == 1 else 'NORMAL'}")
            
            # Analyser les mots les plus influents
            feature_names = self.vectorizer.get_feature_names_out()
            feature_weights = vect.toarray()[0]
            model_coef = self.model.coef_[0]
            
            # Calculer l'influence de chaque mot présent
            word_influences = []
            for i, weight in enumerate(feature_weights):
                if weight > 0:  # Le mot est présent
                    influence = weight * model_coef[i]
                    word_influences.append((feature_names[i], influence))
            
            # Trier par influence (positive = spam, négative = normal)
            word_influences.sort(key=lambda x: abs(x[1]), reverse=True)
            
            print("Mots les plus influents:")
            for word, influence in word_influences[:10]:
                direction = "SPAM" if influence > 0 else "NORMAL"
                print(f"  '{word}': {influence:.3f} -> {direction}")
            
            confidence = max(probabilities)
            return prediction, confidence
            
        except Exception as e:
            print(f"Erreur lors de la prédiction : {str(e)}")
            return None, f"Erreur lors de la prédiction : {str(e)}"
    
    def save_model(self, filepath):
        """Sauvegarde le modèle entraîné"""
        if not self.is_trained:
            return False, "Aucun modèle à sauvegarder"
        
        try:
            model_data = {
                'model': self.model,
                'vectorizer': self.vectorizer,
                'training_accuracy': self.training_accuracy,
                'timestamp': datetime.now()
            }
            joblib.dump(model_data, filepath)
            return True, "Modèle sauvegardé avec succès"
        except Exception as e:
            return False, f"Erreur lors de la sauvegarde : {str(e)}"
    
    def load_model(self, filepath):
        """Charge un modèle pré-entraîné"""
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.vectorizer = model_data['vectorizer']
            self.training_accuracy = model_data['training_accuracy']
            self.is_trained = True
            return True, f"Modèle chargé avec succès!\nPrécision: {self.training_accuracy:.2%}"
        except Exception as e:
            return False, f"Erreur lors du chargement : {str(e)}"

class SpamDetectorGUI:
    def __init__(self):
        self.detector = SpamDetector()
        self.setup_gui()
        # Entraînement automatique au démarrage
        self.root.after(100, self.auto_train_on_startup)
        
    def setup_gui(self):
        """Configure l'interface graphique"""
        self.root = tk.Tk()
        self.root.title("Détecteur de Spam Email - Version Améliorée")
        self.root.geometry("800x600")
        self.root.minsize(600, 400)
        
        # Style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configuration du grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        # Barre de menu
        self.create_menu()
        
        # Statut du modèle
        status_frame = ttk.LabelFrame(main_frame, text="Statut du Modèle", padding="5")
        status_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.status_label = ttk.Label(status_frame, text="Modèle non entraîné", foreground="red")
        self.status_label.pack(side=tk.LEFT)
        
        ttk.Button(status_frame, text="Entraîner le modèle", 
                  command=self.train_model_async).pack(side=tk.RIGHT)
        
        # Zone de saisie
        input_frame = ttk.LabelFrame(main_frame, text="Email à analyser", padding="5")
        input_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        input_frame.columnconfigure(0, weight=1)
        input_frame.rowconfigure(1, weight=1)
        
        ttk.Label(input_frame, text="Collez ou tapez le contenu de l'email ci-dessous :").grid(row=0, column=0, sticky=tk.W)
        
        self.text_input = scrolledtext.ScrolledText(input_frame, wrap=tk.WORD, height=15)
        self.text_input.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(5, 0))
        
        # Boutons d'action
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        self.predict_button = ttk.Button(button_frame, text="Analyser l'Email", 
                                       command=self.predict_email, style="Accent.TButton")
        self.predict_button.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame, text="Effacer", command=self.clear_text).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Charger fichier", command=self.load_file).pack(side=tk.LEFT)
        
        # Zone de résultat
        result_frame = ttk.LabelFrame(main_frame, text="Résultat de l'analyse", padding="10")
        result_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.result_label = ttk.Label(result_frame, text="Aucune analyse effectuée", 
                                    font=("Helvetica", 12, "bold"))
        self.result_label.pack()
        
        self.confidence_label = ttk.Label(result_frame, text="")
        self.confidence_label.pack(pady=(5, 0))
        
        # Barre de progression
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Barre de statut
        self.status_bar = ttk.Label(main_frame, text="Prêt", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
    def create_menu(self):
        """Crée la barre de menu"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # Menu Fichier
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Fichier", menu=file_menu)
        file_menu.add_command(label="Charger modèle", command=self.load_model)
        file_menu.add_command(label="Sauvegarder modèle", command=self.save_model)
        file_menu.add_separator()
        file_menu.add_command(label="Quitter", command=self.root.quit)
        
        # Menu Aide
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Aide", menu=help_menu)
        help_menu.add_command(label="À propos", command=self.show_about)
    
    def auto_train_on_startup(self):
        """Entraîne automatiquement le modèle au démarrage"""
        if os.path.exists("data/combined_data.csv"):
            self.train_model_async()
        else:
            self.status_bar.config(text="Fichier data/combined_data.csv introuvable - Entraînement manuel requis")
    
    def train_model_async(self):
        """Lance l'entraînement du modèle dans un thread séparé"""
        def train():
            self.progress.start()
            self.status_bar.config(text="Entraînement du modèle en cours...")
            self.predict_button.config(state="disabled")
            
            success, message = self.detector.load_and_train_model()
            
            self.progress.stop()
            self.predict_button.config(state="normal")
            
            if success:
                self.status_label.config(text=f"Modèle entraîné (Précision: {self.detector.training_accuracy:.2%})", 
                                       foreground="green")
                self.status_bar.config(text="Modèle prêt à utiliser")
            else:
                self.status_label.config(text="Échec de l'entraînement", foreground="red")
                self.status_bar.config(text="Erreur lors de l'entraînement")
            
            messagebox.showinfo("Entraînement", message)
        
        threading.Thread(target=train, daemon=True).start()
    
    def predict_email(self):
        """Analyse le texte saisi"""
        input_text = self.text_input.get("1.0", tk.END).strip()
        
        if not input_text:
            messagebox.showwarning("Attention", "Veuillez entrer un texte d'email.")
            return
        
        if not self.detector.is_trained:
            messagebox.showerror("Erreur", "Le modèle n'est pas encore entraîné.")
            return
        
        prediction, confidence = self.detector.predict_text(input_text)
        
        if prediction is None:
            messagebox.showerror("Erreur", confidence)
            return
        
        if prediction == 1:
            result_text = "🚨 SPAM / PHISHING DÉTECTÉ"
            result_color = "red"
        else:
            result_text = "✅ EMAIL NORMAL"
            result_color = "green"
        
        self.result_label.config(text=result_text, foreground=result_color)
        self.confidence_label.config(text=f"Niveau de confiance : {confidence:.1%}")
        self.status_bar.config(text="Analyse terminée")
    
    def clear_text(self):
        """Efface le texte saisi"""
        self.text_input.delete("1.0", tk.END)
        self.result_label.config(text="Aucune analyse effectuée", foreground="black")
        self.confidence_label.config(text="")
    
    def load_file(self):
        """Charge le contenu d'un fichier texte"""
        filename = filedialog.askopenfilename(
            title="Charger un fichier email",
            filetypes=[("Fichiers texte", "*.txt"), ("Tous les fichiers", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as file:
                    content = file.read()
                    self.text_input.delete("1.0", tk.END)
                    self.text_input.insert("1.0", content)
                self.status_bar.config(text=f"Fichier chargé : {os.path.basename(filename)}")
            except Exception as e:
                messagebox.showerror("Erreur", f"Impossible de charger le fichier :\n{str(e)}")
    
    def save_model(self):
        """Sauvegarde le modèle entraîné"""
        if not self.detector.is_trained:
            messagebox.showerror("Erreur", "Aucun modèle à sauvegarder")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Sauvegarder le modèle",
            defaultextension=".pkl",
            filetypes=[("Fichiers Pickle", "*.pkl"), ("Tous les fichiers", "*.*")]
        )
        
        if filename:
            success, message = self.detector.save_model(filename)
            if success:
                messagebox.showinfo("Succès", message)
            else:
                messagebox.showerror("Erreur", message)
    
    def load_model(self):
        """Charge un modèle pré-entraîné"""
        filename = filedialog.askopenfilename(
            title="Charger un modèle",
            filetypes=[("Fichiers Pickle", "*.pkl"), ("Tous les fichiers", "*.*")]
        )
        
        if filename:
            success, message = self.detector.load_model(filename)
            if success:
                self.status_label.config(text=f"Modèle chargé (Précision: {self.detector.training_accuracy:.2%})", 
                                       foreground="green")
                messagebox.showinfo("Succès", message)
            else:
                messagebox.showerror("Erreur", message)
    
    def show_about(self):
        """Affiche les informations sur l'application"""
        about_text = """Détecteur de Spam Email - Version Améliorée

Cette application utilise l'apprentissage automatique pour détecter les emails de spam et de phishing.

Fonctionnalités :
• Analyse en temps réel
• Interface utilisateur améliorée
• Sauvegarde/chargement de modèles
• Niveau de confiance des prédictions
• Support des fichiers texte

Développé avec Python, scikit-learn et Tkinter."""
        
        messagebox.showinfo("À propos", about_text)
    
    def run(self):
        """Lance l'application"""
        self.root.mainloop()

# Point d'entrée principal
if __name__ == "__main__":
    try:
        app = SpamDetectorGUI()
        app.run()
    except Exception as e:
        print(f"Erreur lors du lancement de l'application : {e}")
        messagebox.showerror("Erreur critique", f"Impossible de lancer l'application :\n{e}")