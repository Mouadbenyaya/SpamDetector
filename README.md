Projet : Classification des Emails Spam vs Non-Spam
Description
Ce projet consiste à entraîner un modèle de machine learning simple pour détecter si un email est un spam ou non. Le modèle utilise un dataset public d’emails annotés.

Source des données
Les données utilisées proviennent de ce dataset Kaggle :
Email Spam Classification Dataset

Le dataset contient des emails sous forme de texte brut, avec une étiquette 1 pour spam et 0 pour non-spam.

Fonctionnalités
Nettoyage et vectorisation des textes.

Entraînement d’un modèle MultinomialNB pour la classification.

Interface graphique simple développée avec Tkinter pour permettre à l’utilisateur d’entrer un email et d’obtenir immédiatement une prédiction (spam ou non spam).

Interface Tkinter
L’interface graphique comprend :

Une zone de saisie texte pour coller ou écrire un email.

Un bouton "Classifier" qui lance la prédiction.

Une étiquette qui affiche le résultat ("Spam" ou "Non Spam").

Source des données
Les données utilisées pour ce projet proviennent du dataset Email Spam Classification Dataset disponible sur Kaggle :
https://www.kaggle.com/datasets/purusinghvi/email-spam-classification-dataset

Ce dataset contient environ 83 446 emails annotés comme spam (1) ou non spam (0), ce qui permet d’entraîner un modèle fiable de classification.
