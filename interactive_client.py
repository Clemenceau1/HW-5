# ============================================================================
# INTERACTIVE CLIENT PROGRAM - For VS Code Terminal
# ============================================================================
# This script is designed to run in VS Code Terminal (not in notebook cells)
# 
# To run this:
# 1. Make sure all required variables are loaded (run the notebook first, or load the model)
# 2. Open VS Code Terminal
# 3. Run: python interactive_client.py
#
# This will prompt you to enter reviews and determine if they're positive or negative.

import sys
import os

# Try to import required libraries
try:
    import numpy as np
    import pandas as pd
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except ImportError as e:
    print(f"❌ ERROR: Missing required library: {e}")
    print("Please make sure you've installed all requirements and activated your virtual environment.")
    sys.exit(1)

# Check if we're in a notebook environment (shouldn't be, but just in case)
try:
    get_ipython()
    print("⚠️  WARNING: This script is designed for VS Code Terminal, not Jupyter notebooks.")
    print("   For notebooks, use the interactive cell in the notebook instead.")
except NameError:
    pass  # Good, we're in a terminal

def load_model_and_tokenizer():
    """
    Load the trained model and tokenizer.
    This script is designed to be run from VS Code Terminal.
    You need to run it from the notebook first using %run, or save/load the model.
    """
    print("=" * 70)
    print("LOADING MODEL AND TOKENIZER")
    print("=" * 70)
    
    # Try to get from globals if running from notebook with %run
    try:
        import __main__
        if hasattr(__main__, 'model') and hasattr(__main__, 'tokenizer'):
            print("✅ Found model and tokenizer from notebook session!")
            return (__main__.model, __main__.tokenizer, __main__.label_encoder, 
                    __main__.max_length, __main__.remove_stopwords)
    except:
        pass
    
    # Try to get from current module globals (if run with %run)
    try:
        import sys
        if 'model' in globals() and 'tokenizer' in globals():
            print("✅ Found model and tokenizer in current scope!")
            return model, tokenizer, label_encoder, max_length, remove_stopwords
    except:
        pass
    
    # Try to load from saved files
    try:
        import pickle
        from tensorflow import keras
        
        model_path = 'sentiment_model.h5'
        tokenizer_path = 'tokenizer.pkl'
        label_encoder_path = 'label_encoder.pkl'
        
        if os.path.exists(model_path) and os.path.exists(tokenizer_path) and os.path.exists(label_encoder_path):
            print("📂 Loading model and tokenizer from saved files...")
            
            # Load model
            model = keras.models.load_model(model_path)
            print("✅ Model loaded successfully!")
            
            # Load tokenizer
            with open(tokenizer_path, 'rb') as f:
                tokenizer = pickle.load(f)
            print("✅ Tokenizer loaded successfully!")
            
            # Load label encoder
            with open(label_encoder_path, 'rb') as f:
                label_encoder = pickle.load(f)
            print("✅ Label encoder loaded successfully!")
            
            # Load remove_stopwords function (need to recreate it)
            # We'll define it here since it's needed
            import nltk
            from nltk.corpus import stopwords
            import re
            
            # Download NLTK data if needed
            try:
                nltk.download('stopwords', quiet=True)
                nltk.download('punkt', quiet=True)
            except:
                pass
            
            def remove_stopwords(text):
                """Function to remove stop words from text."""
                if not text or (isinstance(text, str) and text.strip() == ''):
                    return ''
                
                # Convert to lowercase
                text = str(text).lower()
                
                # Remove digits
                text = re.sub(r'\d+', '', text)
                
                # Remove special characters (keep only letters and spaces)
                text = re.sub(r'[^a-zA-Z\s]', '', text)
                
                # Tokenize
                words = text.split()
                
                # Get English stop words
                stop_words = set(stopwords.words('english'))
                
                # Define negation words that should NOT be removed
                negation_words = {'not', 'no', 'never', 'neither', 'nobody', 'none', 'nothing', 
                                  'nowhere', 'nor', 'cannot', 'cant', 'dont', 'doesnt', 'didnt',
                                  'wont', 'wouldnt', 'shouldnt', 'couldnt', 'havent', 'hasnt',
                                  'hadnt', 'isnt', 'arent', 'wasnt', 'werent'}
                
                # Remove stop words BUT preserve negation words
                filtered_words = [word for word in words 
                                 if (word not in stop_words or word in negation_words) and len(word) > 1]
                
                # Join words back
                cleaned_text = ' '.join(filtered_words)
                
                return cleaned_text
            
            # max_length should be 120 (from the notebook)
            max_length = 120
            
            print("✅ All components loaded successfully!")
            return model, tokenizer, label_encoder, max_length, remove_stopwords
        else:
            print("\n⚠️  Saved model files not found.")
            missing_files = []
            if not os.path.exists(model_path):
                missing_files.append(model_path)
            if not os.path.exists(tokenizer_path):
                missing_files.append(tokenizer_path)
            if not os.path.exists(label_encoder_path):
                missing_files.append(label_encoder_path)
            print(f"   Missing files: {', '.join(missing_files)}")
    except Exception as e:
        print(f"\n⚠️  Error loading from files: {e}")
    
    print("\n⚠️  NOTE: Model and tokenizer not found.")
    print("\n📝 TO USE THIS SCRIPT:")
    print("=" * 70)
    print("\n   METHOD 1: Save model from notebook, then run this script")
    print("   1. Open sentiment_analysis.ipynb")
    print("   2. Run all cells to train the model")
    print("   3. In a new cell, run the following code to save the model:")
    print("      model.save('sentiment_model.h5')")
    print("      import pickle")
    print("      pickle.dump(tokenizer, open('tokenizer.pkl', 'wb'))")
    print("      pickle.dump(label_encoder, open('label_encoder.pkl', 'wb'))")
    print("   4. Then run this script: python interactive_client.py")
    print("\n   METHOD 2: Run from notebook (Easiest)")
    print("   1. Open sentiment_analysis.ipynb")
    print("   2. Run all cells to train the model")
    print("   3. In a new cell, type: %run interactive_client.py")
    print("   4. The script will use the model from the notebook session")
    print("\n" + "=" * 70)
    return None, None, None, None, None

def predict_sentiment(review_text, model, tokenizer, label_encoder, max_length=120):
    """
    Predict sentiment for a given review text.
    """
    # Step 1: Clean the review from stop words
    cleaned_review = remove_stopwords(review_text)
    
    # Step 2: Convert review into vector
    review_sequence = tokenizer.texts_to_sequences([cleaned_review])
    review_padded = pad_sequences(review_sequence, maxlen=max_length, padding='post', truncating='post')
    
    # Step 3: Use the trained model to predict the sentiment
    prediction_proba = model.predict(review_padded, verbose=0)
    prediction = np.argmax(prediction_proba, axis=1)[0]
    confidence = prediction_proba[0][prediction]
    
    # Map prediction to sentiment label
    sentiment_label = label_encoder.inverse_transform([prediction])[0]
    sentiment_text = 'Positive' if sentiment_label == 1 else 'Negative'
    
    return sentiment_text, confidence

def main():
    """Main interactive client program."""
    print("=" * 70)
    print("INTERACTIVE SENTIMENT ANALYSIS CLIENT PROGRAM")
    print("=" * 70)
    print("\nThis program will analyze your review and determine if it's POSITIVE or NEGATIVE.")
    print("\n" + "=" * 70)
    
    # Load model and tokenizer
    model, tokenizer, label_encoder, max_length, remove_stopwords = load_model_and_tokenizer()
    
    if model is None:
        print("\n" + "=" * 70)
        print("SETUP INSTRUCTIONS")
        print("=" * 70)
        print("\nTo use this script, you need to run it from your Jupyter notebook:")
        print("  1. Open sentiment_analysis.ipynb")
        print("  2. Run all cells to train the model")
        print("  3. In a new cell, type: %run interactive_client.py")
        print("\nOr modify this script to load saved model files.")
        print("=" * 70)
        return
    
    print("\n✅ Model and tokenizer loaded successfully!")
    print("=" * 70)
    
    # Main client program loop
    while True:
        # Step 1: User must prompt a review for a particular product
        print("\n" + "─" * 70)
        print("📍 ENTER YOUR REVIEW:")
        print("─" * 70)
        user_review = input("\n👉 Enter a product review (or 'quit' to exit): ")
        
        # Check for exit command
        if user_review.lower() == 'quit':
            print("\n" + "=" * 70)
            print("Thank you for using the Sentiment Analysis Client Program!")
            print("=" * 70)
            break
        
        # Validate input
        if user_review.strip() == '':
            print("⚠️  Please enter a valid review. Empty reviews are not allowed.")
            continue
        
        # Process the review
        print("\n" + "=" * 70)
        print("PROCESSING REVIEW...")
        print("=" * 70)
        
        # Step 1: User prompts a review for a particular product
        print(f"Step 1 - User Review: {user_review}")
        
        # Step 2: Clean the review from stop words (using the implemented function)
        cleaned_review = remove_stopwords(user_review)
        print(f"Step 2 - Cleaned Review (stop words removed): {cleaned_review}")
        
        # Step 3: Convert the review into a vector (done inside predict_sentiment)
        # Step 4: Use the trained model to predict the sentiment
        # Step 5: Print the predicted sentiment
        sentiment, confidence = predict_sentiment(user_review, model, tokenizer, label_encoder, max_length)
        
        print(f"Step 3 - Review converted to vector (using tokenizer)")
        print(f"Step 4 - Model prediction completed")
        print(f"Step 5 - Predicted Sentiment: {sentiment}")
        print(f"         Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        print("=" * 70)
        
        # Clear summary
        print("\n" + "=" * 70)
        if sentiment == "Positive":
            print("✅ RESULT: This is a POSITIVE review!")
        else:
            print("❌ RESULT: This is a NEGATIVE review!")
        print("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("Program interrupted by user.")
        print("Thank you for using the Sentiment Analysis Client Program!")
        print("=" * 70)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("⚠️  Make sure the model has been trained and all required variables are defined.")
        import traceback
        traceback.print_exc()

