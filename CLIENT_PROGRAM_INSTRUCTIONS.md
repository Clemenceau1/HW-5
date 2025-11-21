# Interactive Client Program - Instructions

## 🎯 Two Ways to Use the Client Program

### ✅ OPTION 1: For Jupyter Notebook / JupyterLab / Google Colab

**Just run Cell 8.2 in the notebook!**

The cell will show an input box where you can type your review interactively.

---

### ✅ OPTION 2: For VS Code Users (Terminal)

VS Code notebooks do **NOT** support interactive `input()` in cells.

**Solution: Use the Terminal version!**

#### Method A: Run from Notebook (Easiest)

1. Open `sentiment_analysis.ipynb`
2. **Run all cells** to train the model
3. In a **new cell**, type:
   ```python
   %run interactive_client.py
   ```
4. The script will use the model from the notebook session
5. Input prompts will work in the notebook output!

#### Method B: Run from VS Code Terminal

1. **First, save the model** (run this in a notebook cell after training):
   ```python
   # Save model and tokenizer
   model.save('sentiment_model.h5')
   import pickle
   pickle.dump(tokenizer, open('tokenizer.pkl', 'wb'))
   pickle.dump(label_encoder, open('label_encoder.pkl', 'wb'))
   ```

2. **Open VS Code Terminal** (Ctrl+` or View → Terminal)

3. **Navigate to project directory:**
   ```bash
   cd "C:\Users\New User\Desktop\HW-5"
   ```

4. **Activate virtual environment** (if using one):
   ```bash
   .venv\Scripts\activate
   ```

5. **Run the script:**
   ```bash
   python interactive_client.py
   ```

6. **Type your reviews** when prompted!

---

## 📝 What the Program Does

1. **Takes your review as input**
2. **Cleans it** (removes stop words using `remove_stopwords()`)
3. **Converts it to a vector** (using tokenizer)
4. **Uses the trained model** to predict sentiment
5. **Tells you if it's POSITIVE ✅ or NEGATIVE ❌**

---

## 🔧 Troubleshooting

**Problem:** "Model and tokenizer not found"

**Solution:** 
- Make sure you've run all cells in the notebook first
- If using Terminal, save the model first (see Method B above)

**Problem:** Input box doesn't appear in VS Code

**Solution:**
- Use Method A (run with `%run` from notebook)
- Or use Method B (run from Terminal)

---

## ✅ All Required Steps Implemented

✅ **Step 1:** User prompts a review for a particular product  
✅ **Step 2:** Clean the review from stop words (using `remove_stopwords()` function)  
✅ **Step 3:** Convert the review into a vector (using tokenizer and pad_sequences)  
✅ **Step 4:** Use the trained model to predict the sentiment  
✅ **Step 5:** Print the predicted sentiment (Positive or Negative) with confidence score

